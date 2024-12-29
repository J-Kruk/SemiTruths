import os
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
from utils import *
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

torch.set_default_tensor_type("torch.cuda.FloatTensor")

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID).to(device)
preprocess = CLIPProcessor.from_pretrained(model_ID)


def main():

    DS = "CelebAHQ"

    formats = {
        "ADE20K": "jpg",
        "CelebAHQ": "jpg",
        "CityScapes": "png",
        "HumanParsing": "png",
        "OpenImages": "jpg",
        "SUN_RGBD": "jpg",
    }

    all_data = {}

    img_sim = []
    cap_sim = []

    CAPTION_DATA_DIR = (
        f"/raid/mphute6/HalfTruths/SemanticDefinition/outputs/consolidated/{DS}.json"
    )
    IMG_DATA_DIR = f"/raid/mphute6/HalfTruths/Half_Truths_Dataset/images/{DS}/"
    FILTER_DATA_DIR = (
        f"/raid/mphute6/HalfTruths/SemanticDefinition/outputs/quality_check/pregen/"
    )

    caption_data_ds = json.load(open(CAPTION_DATA_DIR, "r"))

    for fname in tqdm(caption_data_ds):
        fname_ext = fname + "." + formats[DS]
        while fname_ext[0] == "0":
            fname_ext = fname_ext[1:]
        img_original = Image.open(IMG_DATA_DIR + fname_ext)

        caption_original = caption_data_ds[fname][0]["original_caption"]

        for edit in caption_data_ds[fname]:
            if "quality_flag" in edit:
                continue
            caption_edit = edit["edited_caption"]
            try:
                if prune_length(caption_original, caption_edit):
                    # Calculate similarity
                    img1_cap1_similarity = calculate_image_caption_clip_similarity(
                        img_original, caption_original
                    )
                    img_sim.append(img1_cap1_similarity)

                    cap1_cap2_similarity = calculate_caption_clip_similarity(
                        caption_original, caption_edit
                    )
                    cap_sim.append(cap1_cap2_similarity)

                    edit["img1_cap1_similarity"] = img1_cap1_similarity
                    edit["cap1_cap2_similarity"] = cap1_cap2_similarity

                    all_data[fname_ext] = all_data.get(fname_ext, [])
                    all_data[fname_ext].append(edit)
                else:
                    continue
            except Exception as e:
                print(e)
                continue

    with open(FILTER_DATA_DIR + f"{DS}_1.json", "w") as f:
        json.dump(all_data, f)


if __name__ == "__main__":
    main()
