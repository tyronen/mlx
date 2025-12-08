import math
import os
from contextlib import nullcontext
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model
from models import image_caption_utils
from common import utils


import numpy as np


from collections import defaultdict
from typing import Optional

CLIP = "openai/clip-vit-large-patch14"
VIT = "google/vit-base-patch16-224-in21k"


class ImageDataset(Dataset):
    def get_images(self):
        raise NotImplementedError("Subclasses must implement this method")

    def __init__(
        self,
        file_field,
        caption_field,
        image_dir,
        split="train",
        group_by_image=False,
        precomputed_store=None,
    ):
        self.device = utils.get_device()
        self.file_field = file_field
        self.caption_field = caption_field
        self.image_dir = image_dir
        self.group_by_image = group_by_image
        self.precomputed_store = precomputed_store
        _, unique_images, rows = self.get_images()

        # Split the images (not the individual caption rows) into train/val/test
        np.random.seed(42)  # reproducible splits
        np.random.shuffle(unique_images)

        n_images = len(unique_images)
        train_end = int(0.8 * n_images)
        val_end = int(0.9 * n_images)
        test_end = int(1.0 * n_images)

        if split == "train":
            split_images = set(unique_images[:train_end])
        elif split == "val":
            split_images = set(unique_images[train_end:val_end])
        else:  # "test"
            split_images = set(unique_images[val_end:test_end])

        # Filter rows
        valid_rows = [row for row in rows if row[self.file_field] in split_images]

        if self.group_by_image:
            # Group captions by image filename
            self.grouped_data = defaultdict(list)
            for row in valid_rows:
                self.grouped_data[row[self.file_field]].append(row[self.caption_field])
            # Create list of (filename, [captions])
            self.data = list(self.grouped_data.items())
        else:
            self.captions = valid_rows

        self.tokenizer = image_caption_utils.TOKENIZER
        self.processor = (
            None
            if self.precomputed_store is not None
            else AutoProcessor.from_pretrained(CLIP)
        )

    def __len__(self):
        if self.group_by_image:
            return len(self.data)
        return len(self.captions)

    def process_image(self, img_filename):
        if self.processor is None:
            raise RuntimeError(
                "Processor is unavailable because precomputed features are in use."
            )
        image_path = os.path.join(self.image_dir, img_filename)
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(
                images=image, return_tensors="pt"
            ).pixel_values[0]
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            pixel_values = torch.zeros((3, 224, 224))
        return pixel_values

    def tokenize_caption(self, caption):
        input_ids = self.tokenizer(
            caption,
            max_length=30,
            add_special_tokens=False,
            truncation=True,
        ).input_ids
        input_ids = (
            [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        )
        input_ids.extend([self.tokenizer.pad_token_id] * (32 - len(input_ids)))
        return torch.tensor(input_ids, dtype=torch.long)

    def __getitem__(self, idx):
        if self.group_by_image:
            img_filename, captions = self.data[idx]
            if self.precomputed_store is not None:
                pixel_values = self.precomputed_store.get(img_filename)
            else:
                pixel_values = self.process_image(img_filename)

            # Tokenize all captions for this image
            caption_tensors = [self.tokenize_caption(c) for c in captions]
            # Stack them: [num_captions, L]
            caption_tensors = torch.stack(caption_tensors)

            return pixel_values, caption_tensors
        else:
            row = self.captions[idx]
            img_filename = row[self.file_field]
            if self.precomputed_store is not None:
                pixel_values = self.precomputed_store.get(img_filename)
            else:
                pixel_values = self.process_image(img_filename)
            caption = row[self.caption_field]
            input_tensor = self.tokenize_caption(caption)
            return pixel_values, input_tensor


class Flickr30kDataset(ImageDataset):
    def __init__(self, split="train", precomputed_store=None):
        # Get image dir from util
        image_dir, _, _ = image_caption_utils.get_flickr()
        super().__init__(
            file_field="image",
            caption_field="caption",
            image_dir=image_dir,
            split=split,
            precomputed_store=precomputed_store,
        )

    def get_images(self):
        return image_caption_utils.get_flickr()


class CocoDataset(ImageDataset):
    def __init__(
        self, split="train", use_official_captions=False, precomputed_store=None
    ):
        self.use_official_captions = use_official_captions
        # Get image dir from util
        image_dir, _, _ = image_caption_utils.get_coco(
            use_official_captions=self.use_official_captions
        )
        # If using official captions, we enable grouping optimization
        group_by_image = use_official_captions

        super().__init__(
            file_field="file_name",
            caption_field="text",
            image_dir=image_dir,
            split=split,
            group_by_image=group_by_image,
            precomputed_store=precomputed_store,
        )

    def get_images(self):
        return image_caption_utils.get_coco(
            use_official_captions=self.use_official_captions
        )


class MLPProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ImageEncoder(nn.Module):
    def __init__(self, max_vision_tokens: Optional[int] = 64):
        super().__init__()
        self.device = utils.get_device()
        if max_vision_tokens is not None and max_vision_tokens <= 0:
            max_vision_tokens = None
        self.max_vision_tokens = max_vision_tokens
        self.processor = AutoProcessor.from_pretrained(CLIP, use_fast=False)
        full_model = AutoModel.from_pretrained(
            CLIP,
            use_safetensors=True,
            dtype=torch.bfloat16,
        )
        # Use only the vision model, not the text model
        # On MPS, keep the encoder in float32 to avoid dtype mismatch in MPS matmul.
        vision_dtype = torch.float32 if self.device.type == "mps" else torch.bfloat16
        self.model = full_model.vision_model.to(self.device, dtype=vision_dtype)

        # Freeze the pre-trained weights
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        model_device = next(self.model.parameters()).device
        is_mps_runtime = (model_device.type == "mps") or (self.device.type == "mps")
        autocast_ctx = (
            torch.autocast("cuda") if model_device.type == "cuda" else nullcontext()
        )
        if is_mps_runtime:
            pixel_values = pixel_values.to(model_device, dtype=torch.float32)
        else:
            pixel_values = pixel_values.to(model_device, dtype=torch.bfloat16)
        with torch.no_grad(), autocast_ctx:
            # Return full sequence of patches [B, 257, 1024]
            outputs = self.model(pixel_values=pixel_values).last_hidden_state
        if self.max_vision_tokens and outputs.size(1) > self.max_vision_tokens:
            cls_token = outputs[:, :1, :]
            patch_tokens = outputs[:, 1:, :]
            target_patches = max(0, self.max_vision_tokens - 1)
            if target_patches == 0:
                outputs = cls_token
            else:
                # MPS requires divisible sizes for adaptive pooling; fall back to CPU if needed
                pooling_tokens = patch_tokens
                needs_cpu_pool = (
                    model_device.type == "mps"
                    and patch_tokens.size(1) % target_patches != 0
                )
                if needs_cpu_pool:
                    pooling_tokens = patch_tokens.cpu().float()
                # Adaptive average pooling keeps positional coverage while shrinking sequence length
                pooled = F.adaptive_avg_pool1d(
                    pooling_tokens.transpose(1, 2), target_patches
                )
                if needs_cpu_pool:
                    pooled = pooled.to(patch_tokens.device, dtype=patch_tokens.dtype)
                pooled = pooled.transpose(1, 2)
                outputs = torch.cat([cls_token, pooled], dim=1)
        return outputs


def make_attn_mask(input_ids: torch.Tensor, prefix_len: int = 1):
    """
    Build a 1/0 attention mask that is accepted by Qwen3

    Args:
        input_ids: [B, T] text token ids (no image prefix yet)
        prefix_len: how many prefix tokens (e.g. the projected image patches) are prepended

    Returns:
        mask: [B, prefix_len + T] long tensor, 1 = keep, 0 = pad
    """
    txt_mask = (
        input_ids != image_caption_utils.TOKENIZER.pad_token_id
    ).long()  # 1/0 over text
    prefix = torch.ones(
        input_ids.size(0),
        prefix_len,
        dtype=txt_mask.dtype,
        device=txt_mask.device,
    )
    return torch.cat([prefix, txt_mask], dim=1)


class CombinedTransformer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_decoders: int,
        dropout: float,
        use_mlp_projector: bool = False,
    ):
        super().__init__()

        self.tokenizer = image_caption_utils.TOKENIZER
        # Load in bfloat16 for better numerical stability than fp16
        # BF16 works great on RTX 5090 and uses same memory as FP16
        # Don't use device_map="auto" as it triggers bitsandbytes in PEFT
        device = utils.get_device()
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B-Base",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation=(
                "flash_attention_2" if device.type == "cuda" else "eager"
            ),
        )
        base.config.use_cache = False

        # Qwen‑3 stores its embeddings at base.model.embed_tokens
        base.resize_token_embeddings(len(self.tokenizer))
        base.config.pad_token_id = self.tokenizer.pad_token_id
        self.token_embedding = base.model.embed_tokens
        self.token_embedding.requires_grad_(False)
        for param in base.parameters():
            param.requires_grad = False

        img_proj_out = model_dim
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,  # LoRA rank
            lora_alpha=16,
            lora_dropout=0.05,
        )
        self.decoder = get_peft_model(base, config)
        for name, p in self.decoder.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
        self.token_proj = nn.Identity()
        img_proj_out = base.config.hidden_size

        if use_mlp_projector:
            self.image_projection = MLPProjector(1024, img_proj_out)
        else:
            self.image_projection = nn.Linear(1024, img_proj_out)

    def embed_input_ids(self, input_ids):
        # Create embeddings for the tokens generated so far
        tok_embed = self.token_embedding(input_ids)
        return self.token_proj(tok_embed)

    def decode_image(self, decoder_input, input_ids):
        # Infer prefix length from decoder_input - input_ids length
        # decoder_input: [B, P + T, D]
        # input_ids: [B, T]
        prefix_len = decoder_input.size(1) - input_ids.size(1)
        attn_mask = make_attn_mask(input_ids, prefix_len=prefix_len)

        out = self.decoder(
            inputs_embeds=decoder_input,
            attention_mask=attn_mask,
        )
        return out.logits

    # one autoregressive step for inference
    def decode_step(self, image_features, input_ids):
        """
        Args:
            image_features: [B, P, D] projected image patches (P=257 usually)
            input_ids:     [B, T] tokens generated **so far** (includes BOS, excludes EOS)
        Returns:
            logits for the **next** token: [B, vocab]
        """
        tok_embed = self.embed_input_ids(input_ids)  # [B, T, D]

        # Decoder input = image prefix + *all* tokens generated so far.
        # image_features is [B, P, D] (already projected if passed from generate loop)
        # If it's 2D [B, D], unsqueeze it.
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)

        decoder_input = torch.cat(
            [image_features.to(tok_embed.dtype), tok_embed], dim=1
        )  # [B, P+T, D]

        # Use the same input_ids to build the pad mask (no shifting here)
        full_logits = self.decode_image(decoder_input, input_ids)  # [B, T, vocab]

        # Return logits for the last position (= next token)
        return full_logits[:, -1, :]  # [B, vocab]

    def forward(self, images, input_ids):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)  # [B, L]
        img_encoded = images.to(device)  # [B, P, 1024] or [B, 1024] if old style

        tok_embed = self.embed_input_ids(input_ids)

        # Project images
        img_embed = self.image_projection(img_encoded)  # [B, P, D]

        # If we somehow got 2D images (legacy), fix shape
        if img_embed.dim() == 2:
            img_embed = img_embed.unsqueeze(1)

        # Prepend image embedding to caption embeddings
        decoder_input = torch.cat([img_embed.to(tok_embed.dtype), tok_embed], dim=1)

        # Use input_ids to build the pad mask
        # Return only the logits corresponding to the text tokens
        out = self.decode_image(decoder_input, input_ids)
        return out[:, img_embed.size(1) :, :]
