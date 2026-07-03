"""Frozen CLIP text encoder for VLM haziness captions.

Uses the stock ``transformers`` CLIPModel text encoder (``openai/clip-vit-base-patch32``),
frozen and in eval mode. At training/inference time we tokenize the
cached caption strings and produce per-token features ``(B, N_tokens, D=512)``
which are consumed by the cross-attention block in :mod:`model`.

The encoder is set to ``eval()`` and its parameters are frozen, so it
adds no optimizer state and contributes no gradients. On Colab it
occupies < ~300 MB of VRAM.

Only depends on ``transformers`` (which is already required for the VLM
captioning step), so no additional dependency is needed.
"""
import torch
import torch.nn as nn

CLIP_NAME = "openai/clip-vit-base-patch32"


class CaptionTextEncoder(nn.Module):
    """Frozen CLIP text encoder returning per-token embeddings."""

    def __init__(self, device="cuda", model_name=CLIP_NAME):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        full = CLIPModel.from_pretrained(model_name)
        self.transformer = full.text_model
        self.text_projection = full.text_projection

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        self._device = device
        self.to(device)

        self._tok_cache = {}

    @torch.no_grad()
    def encode_captions(self, captions):
        """Encode a list of caption strings to ``(B, N, 512)`` tokens.

        Runs in fp32 (autocast disabled) so embeddings are stable
        regardless of the caller's mixed-precision context.
        """
        device = self._device
        ids = [self._tok_cache.get(c) for c in captions]
        miss = [c for c, v in zip(captions, ids) if v is None]
        if miss:
            toks = self.tokenizer(miss, padding=True, truncation=True, return_tensors="pt")
            input_ids = toks["input_ids"].to(device)
            for c, row in zip(miss, input_ids):
                if len(self._tok_cache) < 4096:
                    self._tok_cache[c] = row
        # Build padded batch
        rows = []
        for c in captions:
            r = self._tok_cache.get(c)
            if r is None:
                r = self.tokenizer([c], padding=True, truncation=True, return_tensors="pt")["input_ids"][0].to(device)
                if len(self._tok_cache) < 4096:
                    self._tok_cache[c] = r
            rows.append(r)
        input_ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=0).to(device)
        attention_mask = (input_ids != 0).long()

        with torch.autocast(device_type=device, enabled=False):
            last_hidden = self.transformer(input_ids=input_ids,
                                           attention_mask=attention_mask).last_hidden_state  # (B, N, 768)
            x = last_hidden @ self.text_projection                                            # (B, N, 512)
        return x

    @torch.no_grad()
    def encode_captions_tokens(self, input_ids):
        """Encode pre-tokenized ids ``(B, N)`` directly to ``(B, N, 512)``."""
        device = self._device
        input_ids = input_ids.to(device)
        with torch.autocast(device_type=device, enabled=False):
            attention_mask = (input_ids != 0).long()
            last_hidden = self.transformer(input_ids=input_ids,
                                           attention_mask=attention_mask).last_hidden_state
            x = last_hidden @ self.text_projection
        return x