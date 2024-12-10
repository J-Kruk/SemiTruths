"""
This script will run each component in the pipeline to create the Semi-Truths
dataset from a sample of processed semantic segmentation data.

---------------------------------

Copyright (c) 2024 Julia Kruk & Anisha Pal & Mansi Phute & Manogyna Bhattaram

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import pdb
import yaml
import torch
import torchvision

print("torch version:  ", torch.__version__)
print("torchvision version:  ", torchvision.__version__)

import SemiTruths
import SemiTruths.image_augmentation.inpainting.llava_mask_label_pert as mask_pert
import SemiTruths.image_augmentation.inpainting.llava_guided_inpainting as llava_inpaint
from SemiTruths.image_augmentation.prompt_based_editing.prompt_based_image_aug import (
    configure_lance_args,
    create_prompt_based_edits,
)
from utils import load_config_file, merge_args_with_config

# sys.path.append("image_augmentation/")
# sys.path.append("image_augmentation/inpainting")
# sys.path.append("image_augmentation/LLaVA")
# sys.path.append("image_augmentation/LANCE")
import subprocess
import pandas as pd
import argparse
import pdb


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

##################################################################
# Input Data Locations
##################################################################

parser.add_argument(
    "--input_data_pth",
    default="./data/input",
    help="Path to input media.",
)
parser.add_argument(
    "--input_metadata_pth",
    default="./data/input/metadata.csv",
    help="Path to input metadata.",
)

##################################################################
# Inpainting Augmentation Pipeline
##################################################################

parser.add_argument(
    "--llava_model",
    default="liuhaotian/llava-v1.6-mistral-7b",
    help="LLaVA-Mistral huggingface model id.",
)
parser.add_argument(
    "--llava_cache_dir",
    default="./llava_cache",
    help="Directory to store LLaVA cache files.",
)
parser.add_argument(
    "--output_dir_mask_pert",
    default="./data/gen",
    help="Directory to save metadata with perturbed mask labels.",
)
parser.add_argument(
    "--output_dir_img_aug",
    default="./data/gen",
    help="Directory to save metadata with perturbed mask labels.",
)
parser.add_argument(
    "--mask_pert_file",
    default="./data/gen/metadata_pert.json",
    help="Path to json containing label perturbations.",
)
parser.add_argument(
    "--lance_pth",
    default="./data",
    help="Path to where LANCE model will be saved.",
)

###########################################################################
# LANCE: General Parameters
###########################################################################

parser.add_argument(
    "--dataset",
    type=str,
    default="ADE20K",
    help="Dataset to parse",
)
parser.add_argument(
    "--lance_output_path",
    type=str,
    default="outputs",
    help="LANCE output directory",
)
parser.add_argument(
    "--verbose",
    action="store_false",
    help="Logging verbosity",
)

###########################################################################
# LANCE: Caption generator hyperparameters
###########################################################################

parser.add_argument(
    "--gencaption_name",
    type=str,
    help="Captioner name: blip2_opt,blip2_t5,blip2,blip_caption",
    default="blip_caption",
)
parser.add_argument(
    "--load_captions", action="store_true", help="Load captions from path"
)
parser.add_argument(
    "--gencap_dict_path",
    type=str,
    default="outputs/hard_imagenet_captions_blip2.json",
    help="Path to JSON file containing image captions",
)
parser.add_argument(
    "--load_caption_edits", action="store_true", help="Load captions from path"
)
parser.add_argument(
    "--editcap_dict_path",
    type=str,
    default="outputs/hard_imagenet_captions_blip2_edited.json",
    help="Path to JSON file containing edited captions",
)

###########################################################################
# LANCE: Caption editor hyperparameters
###########################################################################

parser.add_argument(
    "--llama_finetuned_path",
    type=str,
    default="./LANCE/checkpoints/caption_editing/lit-llama-lora-finetuned.pth",
    help="Path to finetuning llama model in lightning format",
)
parser.add_argument(
    "--llama_pretrained_path",
    type=str,
    default="/data/jkruk3/half-truths/lance_checkpoints/lit-llama.pth",
    help="Path to pretrained llama model in lightning format",
)
parser.add_argument(
    "--llama_tokenizer_path",
    type=str,
    default="./LANCE/checkpoints/caption_editing/tokenizer.model",
    help="Path to LLAMA tokenizer model",
)
parser.add_argument(
    "--perturbation_type",
    type=str,
    default="all",
    help="Type of perturbation to stress-test against",
)

###########################################################################
# LANCE: Image editing hyperparameters
###########################################################################

parser.add_argument(
    "--ldm_type",
    type=str,
    default="stable_diffusion_v1_4",
    help="Latent Diffusion Model to use",
)
parser.add_argument(
    "--dataset_type",
    type=str,
    help="Dataset loader type: HardImageNet or ImageFolder",
    default="ImageFolder",
)
parser.add_argument(
    "--text_similarity_threshold",
    type=float,
    default=1.0,
    help="Threshold for CLIP text similarity between GT class and word(s) being edited",
)
parser.add_argument(
    "--clip_img_thresh",
    type=float,
    default=0.0,
    help="Threshold for CLIP similarity between original and edited image",
)
parser.add_argument(
    "--clip_dir_thresh",
    type=float,
    default=0.0,
    help="Threshold for CLIP similarity between original and edited direction",
)
parser.add_argument(
    "--clip_thresh",
    type=float,
    default=0.0,
    help="Threshold for CLIP similarity between original and edited image and direction",
)
parser.add_argument(
    "--edit_word_weight",
    type=float,
    default=2.0,
    help="Maximum number of tries for editing a caption",
)
parser.add_argument(
    "--save_inversion",
    action="store_false",
    help="Whether to save image inversion and load from it for future edits",
)
##################################################################

args = parser.parse_args()

config = load_config_file("config.yaml")
args = merge_args_with_config(args, config)

##################################################################
# Diffusion Model Definitions & Keys
##################################################################

diff_models = [
    "StableDiffusion_v4",
    "StableDiffusion_v5",
    "StableDiffusion_XL",
    "Kandinsky_2_2",
    "OpenJourney",
]
diff_models_LANCE = ["SDv4", "SDv5", "OJ"]
file_type_map = {
    "ADE20K": ".jpg",
    "CelebAHQ": ".jpg",
    "CityScapes": ".png",
    "HumanParsing": ".png",
    "OpenImages": ".jpg",
    "SUN_RGBD": ".jpg",
}

##################################################################


# print("STEP (1): Perturbing Mask Labels w/ LLaVA-Mistral-7b for Inpainting")
# mask_pert.perturb_mask_labels(args)


print("\nSTEP (2): Augmenting Images via Diffusion Inpainting")
input_data_ds_qual = llava_inpaint.filter_labels(
    args.mask_pert_file, args.input_metadata_pth
)

for diff_model in diff_models:
    llava_inpaint.prepare_directory_struct(diff_model, args.output_dir_img_aug)
    gen_meta = llava_inpaint.create_img_augmentations(
        input_data_ds_qual,
        args.input_data_pth,
        llava_inpaint.diff_model_dict,
        diff_model,
        args.output_dir_img_aug,
    )

print("\nSTEP (3): Augmenting Images via Prompt-Based Editing")

for i, dm_ in enumerate(diff_models_LANCE):
    print(f"\n     (3.{i+1}) Editing with {dm_}\n")
    for ds in list(file_type_map.keys()):
        subprocess.run(
            [
                "python",
                "-m",
                "SemiTruths.image_augmentation.prompt_based_editing.prompt_based_image_aug_2",
                "--dset_name",
                "ImageFolder",
                "--img_dir",
                args.input_data_pth,
                "--json_path",
                args.input_metadata_pth,
                "--ldm_type",
                dm_,
                "--lance_output_path",
                args.lance_output_path,
                "--dataset",
                ds,
                "--editcap_dict_path",
                "./data/editcap_dict.json",
                "--gencap_dict_path",
                "./data/prompt-prompt/gencap_dict.json",
            ],
        )

# for i, dm_ in enumerate(diff_models_LANCE):
#     print(f"\n     (3.{i+1}) Editing with {dm_}\n")
#     for ds in list(file_type_map.keys()):

#         # create LANCE args:
#         lance_args = configure_lance_args(parser, args, dm_, ds)
#         create_prompt_based_edits(lance_args)

print(f"\nSemi-Truths pipeline is complete! Data saved at {args.output_dir_img_aug}")
