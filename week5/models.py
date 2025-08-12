import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
import utils

CLIP = "openai/clip-vit-base-patch32"
VIT = "google/vit-base-patch16-224-in21k"
IMAGES_PATH = "data/image_features.pt"

import numpy as np


class Flickr30kDataset(Dataset):
    def __init__(self, split="train"):
        _, unique_images, rows = utils.get_captions()

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

        # Keep only caption rows whose image belongs to the chosen split
        self.captions = [row for row in rows if row["image"] in split_images]

        self.tokenizer = utils.TOKENIZER
        self.image_features = torch.load(IMAGES_PATH, weights_only=False)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        row = self.captions[idx]
        img_filename = row["image"]
        image = torch.tensor(self.image_features[img_filename])
        caption = row["caption"]
        # Pre‑tokenize caption once (LongTensor [L])
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
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        return image, input_tensor


def attention(k_dim, q, k, v, mask_tensor):
    kt = k.transpose(-2, -1)
    # do attention(Q, K, V) = softmax(Q·K^T / sqrt(dims))·V to get hidden state (where · is dot product)
    attn_dot_product = torch.matmul(q, kt)
    attn_scaled = attn_dot_product / math.sqrt(k_dim)
    if mask_tensor is not None:
        attn_scaled = attn_scaled.masked_fill(mask_tensor, -torch.inf)
    attn_probs = torch.softmax(attn_scaled, dim=-1)
    return torch.matmul(attn_probs, v)


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.k_dim = model_dim // num_heads
        self.wqkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.endmulti = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def rearrange(self, vector, B, L):
        return vector.reshape(B, L, self.num_heads, self.k_dim).transpose(1, 2)

    def forward(self, x, attn_mask):
        B, L, D = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split(self.model_dim, dim=-1)
        qh = self.rearrange(q, B, L)
        kh = self.rearrange(k, B, L)
        vh = self.rearrange(v, B, L)

        mask_tensor = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        # Expand to 1×1×L×L so it can broadcast with per‑batch pad masks and per‑head scores
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        pad_mask = (attn_mask == 0).unsqueeze(1).unsqueeze(2)
        mask_tensor = mask_tensor | pad_mask

        attended = attention(self.k_dim, qh, kh, vh, mask_tensor=mask_tensor)
        concatted = attended.transpose(1, 2).reshape(B, L, self.model_dim)
        concatted = self.dropout(concatted)
        return self.endmulti(concatted)


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(model_dim, ffn_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim, bias=True),
        )

    def forward(self, x):
        return self.sequence(x)


class Decoder(nn.Module):
    def __init__(self, model_dim: int, ffn_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.masked_self_mha = SelfAttention(model_dim=model_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim=model_dim, ffn_dim=ffn_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, attn_mask):
        stage1 = self.masked_self_mha(data, attn_mask)
        addnormed_text = self.norm1(data + self.dropout(stage1))
        ffned = self.ffn(addnormed_text)
        return self.norm2(addnormed_text + self.dropout(ffned))


class CustomDecoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_decoders: int,
        dropout: float,
        vocab_size: int,
    ):
        super().__init__()

        def make_decoder() -> nn.Module:
            return Decoder(
                model_dim=model_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        self.decoder_series = nn.ModuleList(
            [make_decoder() for _ in range(num_decoders)]
        )
        self.linear = nn.Linear(model_dim, vocab_size)

    def forward(self, input, attn_mask):
        for decoder in self.decoder_series:
            input = decoder(input, attn_mask)

        return self.linear(input)  # [B, L-1, vocab]


class VitEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = utils.get_device()
        self.vit_processor = AutoProcessor.from_pretrained(VIT, use_fast=False)
        self.vit_model = AutoModel.from_pretrained(VIT, use_safetensors=True).to(
            self.device
        )

        # Freeze the pre-trained weights
        for param in self.vit_model.parameters():
            param.requires_grad = False

    def forward(self, images):
        inputs = self.vit_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vit_model(**inputs).last_hidden_state.mean(dim=1)
        return outputs


def make_attn_mask(input_ids: torch.Tensor):
    """
    Build a 1/0 attention mask that is accepted by both Qwen3 and our custom decoder.

    Args:
        input_ids: [B, T] text token ids (no image prefix yet)
        pad_id: tokenizer.pad_token_id (or eos if PAD is shared)
        extra_prefix: how many prefix tokens (e.g. the projected image) are prepended

    Returns:
        mask: [B, extra_prefix + T] long tensor, 1 = keep, 0 = pad
    """
    txt_mask = (input_ids != utils.TOKENIZER.pad_token_id).long()  # 1/0 over text
    prefix = torch.ones(
        input_ids.size(0),
        1,
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
        use_custom_decoder: bool,
    ):
        super().__init__()

        self.tokenizer = utils.TOKENIZER
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,           # typical defaults
            llm_int8_has_fp16_weight=False,
        )
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B-Base", trust_remote_code=True,     
            quantization_config=bnb_config,
            device_map="auto",
        )
        base.gradient_checkpointing_enable()
        base.config.use_cache = False

        # Qwen‑3 stores its embeddings at base.model.embed_tokens
        self.token_embedding = base.model.embed_tokens
        self.token_embedding.requires_grad_(False)
        self.use_custom_decoder = use_custom_decoder
        for param in base.parameters():
            param.requires_grad = False

        img_proj_out = model_dim
        if use_custom_decoder:
            self.decoder = CustomDecoder(
                model_dim=model_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                num_decoders=num_decoders,
                dropout=dropout,
                vocab_size=self.token_embedding.num_embeddings,
            )
            self.token_proj = nn.Linear(self.token_embedding.embedding_dim, model_dim)
        else:
            config = LoraConfig(
               task_type="CAUSAL_LM",
                r=8,                # LoRA rank
                lora_alpha=16,
                lora_dropout=0.05,
            )
            self.decoder = get_peft_model(base, config)
            for name, p in self.decoder.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True
            self.token_proj = nn.Identity()
            img_proj_out = self.decoder.config.hidden_size
        self.image_projection = nn.Linear(768, img_proj_out)

    def embed_input_ids(self, input_ids):
        # Create embeddings for the tokens generated so far
        tok_embed = self.token_embedding(input_ids)
        return self.token_proj(tok_embed)

    def decode_image(self, decoder_input, input_ids):
        attn_mask = make_attn_mask(input_ids)
        if self.use_custom_decoder:
            return self.decoder(decoder_input, attn_mask)

        out = self.decoder(
            inputs_embeds=decoder_input,
            attention_mask=attn_mask,
        )
        return out.logits

    # one autoregressive step for inference 
    def decode_step(self, image_features, input_ids):
        """
        Args:
            image_features: [B, D] projected image vector (same D as decoder hidden)
            input_ids:     [B, T] tokens generated **so far** (includes BOS, excludes EOS)
        Returns:
            logits for the **next** token: [B, vocab]
        """
        tok_embed = self.embed_input_ids(input_ids)  # [B, T, D]

        # Decoder input = image prefix + *all* tokens generated so far.
        images = image_features.unsqueeze(1).to(tok_embed.dtype)
        decoder_input = torch.cat([images, tok_embed], dim=1)  # [B, 1+T, D]

        # Use the same input_ids to build the pad mask (no shifting here)
        full_logits = self.decode_image(decoder_input, input_ids)  # [B, T, vocab]

        # Return logits for the last position (= next token)
        return full_logits[:, -1, :]  # [B, vocab]

    def forward(self, images, input_ids):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)  # [B, L]
        img_encoded = images.to(device)

        tok_embed = self.embed_input_ids(input_ids)

        # Encode image
        img_embed = self.image_projection(img_encoded).unsqueeze(1)  # [B, 1, D]

        # Prepend image embedding to caption embeddings
        decoder_input = torch.cat([img_embed.to(tok_embed.dtype), tok_embed[:, :-1, :]], dim=1)

        # Use input_ids without the last token so the mask length matches decoder_input (image + L‑1 text tokens)
        return self.decode_image(decoder_input, input_ids[:, :-1])
