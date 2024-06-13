from __future__ import annotations
import os
import sys
import warnings
import difflib
import pandas as pd
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# pd.set_option("display.max_colwidth", 40)

from IPython.display import HTML, Markdown, display
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from nltk.stem import WordNetLemmatizer

###########################################################
# Miscellaneous
###########################################################


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def path_to_image_html(path):
    return '<img src="' + os.path.join(os.getcwd(), path) + '" width="512" >'

def print_header(header):
    return Markdown(header)

def render_as_html(df, sort_by: Optional[list] = None):
    df = df[
        [
            "Image",
            "Label",
            "Prediction",
            "Caption",
            "Edited Caption",            
            # "Reconstruction",
            # "Modification",
            "Edit Type",
            "LANCE",
            "LANCE prediction",
            "Sensitivity",
            # "Avg. sensitivity",
            # "Cluster Name",
        ]
    ]
    df = df.sort_values(by=sort_by, ascending=False)

    image_cols = [
        "Image",
        "Reconstruction",
        "LANCE",
    ]  # <- define which columns will be used to convert to html
    # Create the dictionariy to be passed as formatters
    format_dict = {}
    for image_col in image_cols:
        format_dict[image_col] = path_to_image_html
    df.style.set_properties(**{"text-align": "center"})
    html_ = df.to_html(escape=False, formatters=format_dict).replace(
        "<td>", '<td align="center">'
    ).replace("<th>", '<th style="text-align:center;">')
    return html_


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

        warnings.filterwarnings("ignore", category=DeprecationWarning)

###########################################################
# Plotting
###########################################################
def plot_sensitivity(df, model_name, cls_name, x="", y="", sort_by=[]):    
    sns.set(style="darkgrid")
    plt.figure(figsize=(6, 4))
    if sort_by:
        df = df.sort_values(by=sort_by, ascending=False)

    sns.boxplot(data=df, x=x, y=y, flierprops={"markersize": 1}, showfliers=False)
    plt.xticks(fontsize=12, rotation=45)
    # Add horizontal line
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('{} sensitivity for "{}"'.format(model_name, cls_name), fontsize=16)
    plt.ylabel("{}".format(y), fontsize=16)
    plt.xlabel(x, fontsize=16)
    plt.show()

###########################################################
# Similarity utils
###########################################################
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
    def encode_image(
        self, image: torch.Tensor
    ) -> torch.Tensor:  # Input images in range [0, 1].
        image = F.interpolate(
            image.float(), size=self.size, mode="bicubic", align_corners=False
        )
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    @torch.no_grad()
    def text_similarity(
        self,
        text_0: list[str],
        text_1: list[str],
        get_feats: Optional[bool] = False,
        lemmatize: Optional[bool] = False,
    ) -> torch.Tensor:
        if lemmatize:
            text_0 = [self.lemmatizer.lemmatize(t0) for t0 in text_0]
            text_1 = [self.lemmatizer.lemmatize(t1) for t1 in text_1]

        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim = text_features_0 @ text_features_1.T
        if get_feats:
            return sim, text_features_0, text_features_1
        return sim

    @torch.no_grad()
    def forward(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        text_0: list[str],
        text_1: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(
            image_features_1 - image_features_0, text_features_1 - text_features_0
        )
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        return sim_0, sim_1, sim_direction, sim_image

    @torch.no_grad()
    def pred_consistency(
        self, image: torch.Tensor, text_0: list[str], text_1: list[str]
    ) -> bool:
        sim_0, sim_1, _, _ = self.forward(image, image, text_0, text_1)
        return (sim_1 > sim_0)[0].item()

def compute_diff(cap1, cap2):
        """
        Computes the difference between two captions.
        """
        words_changed = list(
            difflib.ndiff(
                cap1.lower().replace(".", "").strip().split(" "),
                cap2.lower().replace(".", "").strip().split(" "),
            )
        )
        diff1, diff1_len, maxdiff1, maxdiff1_len = [], 0, [], 0
        diff2, diff2_len, maxdiff2, maxdiff2_len = [], 0, [], 0
        for word in words_changed:
            if "+" in word and len(word) > 2:
                diff1.append(word[2:])
                diff1_len += 1
                if diff1_len > maxdiff1_len:
                    maxdiff1_len = diff1_len
                    maxdiff1 = diff1
            else:
                diff1_len = 0
                diff1 = []

            if "-" in word and len(word) > 2:
                diff2.append(word[2:])
                diff2_len += 1
                if diff2_len > maxdiff2_len:
                    maxdiff2_len = diff2_len
                    maxdiff2 = diff2
            else:
                diff2_len = 0
                diff2 = []

        if len(maxdiff1):
            edit_word = " ".join(maxdiff1)
        else:
            edit_word = ""

        if len(maxdiff2):
            original_word = " ".join(maxdiff2)
        else:
            original_word = ""

        return original_word, edit_word