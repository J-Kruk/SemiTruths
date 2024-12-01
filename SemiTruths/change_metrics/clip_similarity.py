from nltk.stem import WordNetLemmatizer
import warnings
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nltk.stem import WordNetLemmatizer
from typing import Optional

class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

with warnings.catch_warnings():
    with SuppressPrint():
        import clip

class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14", device: str = "cuda"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {
            "RN50x4": 288,
            "RN50x16": 384,
            "RN50x64": 448,
            "ViT-L/14@336px": 336,
        }.get(name, 224)
        self.device = device

        self.model, _ = clip.load(
            name, device=self.device, download_root="./checkpoints"
        )
        self.model.eval().requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)).to(self.device)
        )
        self.register_buffer(
            "std", torch.tensor((0.26862954, 0.26130258, 0.27577711)).to(self.device)
        )
        self.lemmatizer = WordNetLemmatizer()

    @torch.no_grad()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    @torch.no_grad()
    def text_similarity(
        self,
        text_0: list[str],
        text_1: list[str],
        lemmatize: Optional[bool] = False,
    ) -> torch.Tensor:
        if lemmatize:
            text_0 = [self.lemmatizer.lemmatize(t0) for t0 in text_0]
            text_1 = [self.lemmatizer.lemmatize(t1) for t1 in text_1]

        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim = text_features_0 @ text_features_1.T
        return sim