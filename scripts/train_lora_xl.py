#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion XL for text2image with support for LoRA."""
from dataclasses import dataclass, field
import itertools
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict
import tempfile
from PIL import Image
import wandb
from collections import defaultdict

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from ddpo_pytorch.diffusers_patch.xl_pipeline_with_logprob import xl_pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import ddpo_pytorch.rewards

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    dataset_name=str,
    repo_folder=None,
    vae_path=None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
dataset: {dataset_name}
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}

Special VAE used for training: {vae_path}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_vae_model_name_or_path: str = None
    revision: str = None


@dataclass
class DatasetConfig:
    caption_file: str = "ddpo_pytorch/assets/paintings.txt"
    # TODO Do validation
    #validation_count: int = 10


@dataclass
class TrainingConfig:
    output_dir: str = field(
        default="sd-model-finetuned-lora",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    cache_dir: str = field(
        default=None,
        metadata={
            "help": "The directory where the downloaded models and datasets will be stored."
        },
    )
    seed: int = field(
        default=None, metadata={"help": "A seed for reproducible training."}
    )
    resolution: int = field(
        default=512,
        metadata={
            "help": "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"
        },
    )
    num_epochs: int = field(
        default=100, metadata={"help": "Number of training epochs."}
    )
    checkpointing_steps: int = field(
        default=10,
        metadata={
            "help": "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint, and are also suitable for resuming training using `--resume_from_checkpoint`."
        },
    )
    checkpoints_total_limit: int = field(
        default=5, metadata={"help": "Max number of checkpoints to store."}
    )
    resume_from_checkpoint: str = field(
        default=None,
        metadata={
            "help": "Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or 'latest' to automatically select the last available checkpoint."
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={
            "help": "Initial learning rate (after the potential warmup period) to use."
        },
    )
    scale_lr: bool = field(
        default=False,
        metadata={
            "help": "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size."
        },
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={
            "help": 'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
        },
    )
    lr_warmup_steps: int = field(
        default=500,
        metadata={"help": "Number of steps for the warmup in the lr scheduler."},
    )
    snr_gamma: float = field(
        default=None,
        metadata={
            "help": "SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556."
        },
    )
    allow_tf32: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        },
    )
    use_8bit_adam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use 8-bit Adam from bitsandbytes."},
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "The beta1 parameter for the Adam optimizer."}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "The beta2 parameter for the Adam optimizer."}
    )
    adam_weight_decay: float = field(
        default=1e-2, metadata={"help": "Weight decay to use."}
    )
    adam_epsilon: float = field(
        default=1e-08, metadata={"help": "Epsilon value for the Adam optimizer."}
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to push the model to the Hub."}
    )
    hub_token: str = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )
    prediction_type: str = field(
        default=None,
        metadata={
            "help": "The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen."
        },
    )
    hub_model_id: str = field(
        default=None,
        metadata={
            "help": "The name of the repository to keep in sync with the local `output_dir`."
        },
    )
    logging_dir: str = field(
        default="logs",
        metadata={
            "help": "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        },
    )
    report_to: str = field(
        default="tensorboard",
        metadata={
            "help": 'The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        },
    )
    mixed_precision: str = field(
        default=None,
        metadata={
            "help": "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        },
    )

    local_rank: int = field(
        default=-1,
        metadata={
            "help": "For distributed training: local_rank",
        },
    )
    enable_xformers_memory_efficient_attention: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use xformers.",
        },
    )
    noise_offset: float = field(
        default=0,
        metadata={
            "help": "The scale of noise offset.",
        },
    )
    rank: int = field(
        default=4,
        metadata={
            "help": "The dimension of the LoRA update matrices.",
        },
    )

    sample_num_steps: int = field(
        default=20, metadata={"help": "Number of steps to sample during sampling phase"}
    )
    sample_batch_size: int = field(
        default=8, metadata={"help": "Batch size of sampling, during sampling phase"}
    )
    sample_num_batches_per_epoch: int = field(
        default=1,
        metadata={
            "help": "number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch * batch_size * num_gpus`."
        },
    )
    sample_guidance_scale: float = field(
        default=6.5, metadata={"help": "cfg param for sampling phase"}
    )
    sample_eta: float = field(
        default=1.0,
        metadata={
            "help": "eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0 being fully deterministic and 1.0 being equivalent to the DDPM sampler."
        },
    )

    train_adv_clip_max: float = field(
        default=5,
        metadata={
            "help": "clip advantages to the range [-adv_clip_max, adv_clip_max]."
        },
    )
    train_clip_range: float = field(
        default=1e-4, metadata={"help": "the PPO clip range."}
    )
    reward_fn: str = field(
        default="hpsv2_score",
        metadata={
            "help": "reward function to use. see `rewards.py` for available reward functions."
        },
    )

    train_batch_size: int = field(
            default=1,
            metadata={"help": "per device batch size for train phase"})

    train_num_inner_epochs: int = field(
            default = 1,
            metadata={"help": "number of epochs to train through the sampled data from the outer phase"})


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    """
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[
                f"{attn_processor_key}.{parameter_key}"
            ] = parameter

    return attn_processors_state_dict


def tokenize_prompt(tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompts, ):
    prompt_embeds_list = []
    pooled_prompt_embeds=None
    bs_embed = None

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        inputs = tokenize_prompt(tokenizer, prompts)
        inputs = {k:v.to(text_encoder.device) for k,v in inputs.items()}

        outputs = text_encoder(**inputs, output_hidden_states=True)

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = outputs[0]

        prompt_embeds = outputs.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    assert pooled_prompt_embeds is not None
    assert bs_embed is not None

    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def main(model_config: ModelConfig, dataset_config: DatasetConfig, training_config: TrainingConfig):
    logging_dir = Path(training_config.output_dir, training_config.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=training_config.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        mixed_precision=training_config.mixed_precision,
        log_with=training_config.report_to,
        project_config=accelerator_project_config,
    )

    if training_config.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if training_config.seed is not None:
        set_seed(training_config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if training_config.output_dir is not None:
            os.makedirs(training_config.output_dir, exist_ok=True)

        if training_config.push_to_hub:
            repo_id = create_repo(
                repo_id=training_config.hub_model_id or Path(training_config.output_dir).name,
                exist_ok=True,
                token=training_config.hub_token,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        model_config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=model_config.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        model_config.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=model_config.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        model_config.pretrained_model_name_or_path, model_config.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        model_config.pretrained_model_name_or_path, model_config.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDIMScheduler.from_pretrained(
        model_config.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        model_config.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=model_config.revision,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        model_config.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=model_config.revision,
    )
    vae_path = (
        model_config.pretrained_model_name_or_path
        if model_config.pretrained_vae_model_name_or_path is None
        else model_config.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if model_config.pretrained_vae_model_name_or_path is None else None,
        revision=model_config.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_config.pretrained_model_name_or_path, subfolder="unet", revision=model_config.revision
    )

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    unet.to(accelerator.device, dtype=weight_dtype)
    if model_config.pretrained_vae_model_name_or_path is None:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if training_config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    unet_lora_attn_procs = {}
    unet_lora_parameters = []
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_processor_class = (
            LoRAAttnProcessor2_0
            if hasattr(F, "scaled_dot_product_attention")
            else LoRAAttnProcessor
        )
        module = lora_attn_processor_class(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=training_config.rank,
        )
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = unet_attn_processors_state_dict(model)
                elif isinstance(
                    model, type(accelerator.unwrap_model(text_encoder_one))
                ):
                    text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(
                        model
                    )
                elif isinstance(
                    model, type(accelerator.unwrap_model(text_encoder_two))
                ):
                    text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(
                        model
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(
            lora_state_dict, network_alphas=network_alphas, unet=unet_
        )

        text_encoder_state_dict = {
            k: v for k, v in lora_state_dict.items() if "text_encoder." in k
        }
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_state_dict,
            network_alphas=network_alphas,
            text_encoder=text_encoder_one_,
        )

        text_encoder_2_state_dict = {
            k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k
        }
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_2_state_dict,
            network_alphas=network_alphas,
            text_encoder=text_encoder_two_,
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if training_config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if training_config.scale_lr:
        training_config.learning_rate = (
            training_config.learning_rate
            * training_config.gradient_accumulation_steps
            * training_config.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if training_config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = unet_lora_parameters
    optimizer = optimizer_class(
        params_to_optimize,
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        weight_decay=training_config.adam_weight_decay,
        eps=training_config.adam_epsilon,
    )

    # Get the reward
    reward_fn = getattr(ddpo_pytorch.rewards, training_config.reward_fn)()

    # Get the captions
    path = dataset_config.caption_file
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
    with open(path, "r") as f:
        captions = [line.strip() for line in f.readlines()]

    dataset = datasets.Dataset.from_dict({'caption': captions})
    # TODO split
    train_dataset=dataset

    # Prepare everything with our `accelerator`.
    unet, optimizer, = accelerator.prepare(
        unet, optimizer
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(training_config))

    samples_per_epoch = training_config.sample_batch_size * accelerator.num_processes * training_config.sample_num_batches_per_epoch
    total_train_batch_size = (
        training_config.train_batch_size * accelerator.num_processes * training_config.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {training_config.num_epochs}")
    logger.info(f"  Sample batch size per device = {training_config.sample_batch_size}")
    logger.info(f"  Train batch size per device = {training_config.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_config.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {training_config.train_num_inner_epochs}")

    assert training_config.sample_batch_size >= training_config.train_batch_size
    assert training_config.sample_batch_size % training_config.train_batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0


    # Potentially load in the weights and states from a previous save
    if training_config.resume_from_checkpoint:
        if training_config.resume_from_checkpoint != "latest":
            path = os.path.basename(training_config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(training_config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{training_config.resume_from_checkpoint}' does not exist. "
            )
            raise ValueError(
f"Checkpoint '{training_config.resume_from_checkpoint}' does not exist. "
                    )
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(training_config.output_dir, path))
            import bpdb
            bpdb.set_trace()
            # TODO the path might be broken
            first_epoch = int(training_config.output_dir.split("-")[-1]) + 1
    else:
        first_epoch = 0

    neg_prompt_embeds, pooled_neg_prompt_embeds = encode_prompt(
                text_encoders=[text_encoder_one, text_encoder_two],
                tokenizers=[tokenizer_one, tokenizer_two],
                prompts=[""],
        )

    neg_prompt_embeds = neg_prompt_embeds.repeat(training_config.sample_batch_size, 1, 1)
    pooled_neg_prompt_embeds = pooled_neg_prompt_embeds.repeat(training_config.sample_batch_size, 1)


    global_step = 0
    for epoch in range(first_epoch, training_config.num_epochs):

        #################### SAMPLING ####################
        unet.eval()
        samples = []
        last_images = []
        last_rewards = None
        last_prompts = []

        for i in tqdm(
            range(training_config.sample_num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):

            prompts = train_dataset.shuffle()['caption'][:training_config.sample_batch_size]

            last_prompts = prompts

            # encode prompts
            prompt_ids = tokenizer_one
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders=[text_encoder_one, text_encoder_two],
                tokenizers=[tokenizer_one, tokenizer_two],
                prompts=prompts,
            )


            # TODO should this be constructed in the inner loop or is this slow?
            print("creating pipeline...")
            pipeline = StableDiffusionXLPipeline(
                    vae = vae,
                    text_encoder=text_encoder_two,
                    text_encoder_2=text_encoder_two,
                    tokenizer=tokenizer_one,
                    tokenizer_2=tokenizer_two,
                    unet=unet,
                    scheduler=noise_scheduler,
                )
            print("done creating pipeline.")

            images, latents, log_probs = xl_pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=neg_prompt_embeds,
                    negative_pooled_prompt_embeds=pooled_neg_prompt_embeds,
                    num_inference_steps=training_config.sample_num_steps,
                    guidance_scale=training_config.sample_guidance_scale,
                    eta=training_config.sample_eta,
                    output_type="pt",
                    height=512,
                    width=512,
                )

            last_images = images

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
            timesteps = pipeline.scheduler.timesteps.repeat(training_config.sample_batch_size, 1)  # (batch_size, num_steps)

            rewards, _ = reward_fn(images, prompts, {})
            rewards = torch.as_tensor(rewards, device=accelerator.device)

            last_rewards = rewards

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # Save generated images
        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(last_images):
                pil = Image.fromarray((image.cpu().to(torch.float32).numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                #pil = pil.resize((512, 512))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            accelerator.log(
                {
                    "images": [
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt} | {reward:.2f}")
                        for i, (prompt, reward) in enumerate(zip(last_prompts, last_rewards))  # only log rewards from process 0
                    ],
                },
                step=global_step,
            )

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        # log rewards and images
        accelerator.log(
            {"reward": rewards, "epoch": epoch, "reward_mean": rewards.mean(), "reward_std": rewards.std()},
            step=global_step,
        )

        
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(accelerator.num_processes, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == training_config.sample_batch_size * training_config.sample_num_batches_per_epoch
        assert num_timesteps == training_config.sample_num_steps


        #################### TRAINING ####################
        for inner_epoch in range(training_config.train_num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]

            # rebatch for training
            samples_batched = {k: v.reshape(-1, training_config.train_batch_size, *v.shape[1:]) for k, v in samples.items()}

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            # train
            unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if training_config.sample_guidance_scale:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat([neg_prompt_embeds, sample["prompt_embeds"]])
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(num_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        if training_config.sample_guidance_scale:
                            noise_pred = unet(
                                torch.cat([sample["latents"][:, j]] * 2),
                                torch.cat([sample["timesteps"][:, j]] * 2),
                                embeds,
                            ).sample
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + training_config.sample_guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )
                        else:
                            noise_pred = unet(
                                sample["latents"][:, j],
                                sample["timesteps"][:, j],
                                embeds,
                            ).sample
                        # compute the log prob of next_latents given latents under the current model
                        _, log_prob = ddim_step_with_logprob(
                            noise_scheduler,
                            noise_pred,
                            sample["timesteps"][:, j],
                            sample["latents"][:, j],
                            eta=training_config.sample_eta,
                            prev_sample=sample["next_latents"][:, j],
                        )

                        # ppo logic
                        advantages = torch.clamp(
                            sample["advantages"], -training_config.train_adv_clip_max, training_config.train_adv_clip_max
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio, 1.0 - training_config.train_clip_range, 1.0 + training_config.train_clip_range
                        )
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        # debugging values
                        # John Schulman says that (ratio - 1) - log(ratio) is a better
                        # estimator, but most existing code uses this so...
                        # http://joschu.net/blog/kl-approx.html
                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > training_config.train_clip_range).float()))
                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_timesteps - 1) and (
                            i + 1
                        ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if epoch != 0 and epoch % training_config.checkpointing_steps == 0 and accelerator.is_main_process:
            accelerator.save_state()


    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet_lora_layers = unet_attn_processors_state_dict(unet)

        text_encoder_lora_layers = None
        text_encoder_2_lora_layers = None

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=training_config.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

        del unet
        del text_encoder_one
        del text_encoder_two
        del text_encoder_lora_layers
        del text_encoder_2_lora_layers
        torch.cuda.empty_cache()

        # Final inference
        # Load previous pipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_config.pretrained_model_name_or_path,
            vae=vae,
            revision=model_config.revision,
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.load_lora_weights(training_config.output_dir)

        if training_config.push_to_hub:
            repo_id = create_repo(
                repo_id=training_config.hub_model_id or Path(training_config.output_dir).name, exist_ok=True, token=training_config.hub_token
            ).repo_id

            save_model_card(
                repo_id,
                base_model=model_config.pretrained_model_name_or_path,
                repo_folder=training_config.output_dir,
                vae_path=model_config.pretrained_vae_model_name_or_path,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=training_config.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
