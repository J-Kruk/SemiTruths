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
import torch
import torchvision

print("torch version:  ", torch.__version__)
print("torchvision version:  ", torchvision.__version__)

import SemiTruths
import SemiTruths.image_augmentation.inpainting.llava_mask_label_pert as mask_pert
import SemiTruths.image_augmentation.inpainting.llava_guided_inpainting as llava_inpaint

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
    "--output_dir_pert",
    default="./data",
    help="Directory to save metadata with perturbed mask labels.",
)
parser.add_argument(
    "--output_dir_aug",
    default="./data/gen",
    help="Directory to save metadata with perturbed mask labels.",
)
parser.add_argument(
    "--pert_file",
    default="./data/metadata_pert.json",
    help="Path to json containing label perturbations.",
)
parser.add_argument(
    "--lance_pth",
    default="./data",
    help="Path to where LANCE model will be saved.",
)
parser.add_argument(
    "--qual_output_path",
    default="./data/postgen_quality_check.csv",
    help="Path to output quality checked metadata.",
)
args = parser.parse_args()

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


# print("STEP (1): Perturbing Mask Labels w/ LLaVA-Mistral-7b for Inpainting")
# mask_pert.perturb_mask_labels(args)


# print("\nSTEP (2): Augmenting Images via Diffusion Inpainting")
# input_data_ds_qual = llava_inpaint.filter_labels(args.pert_file, args.input_metadata_pth)

# for diff_model in diff_models:
#     llava_inpaint.prepare_directory_struct(diff_model, args.output_dir_aug)
#     gen_meta = llava_inpaint.create_img_augmentations(
#         input_data_ds_qual,
#         args.input_data_pth,
#         llava_inpaint.diff_model_dict,
#         diff_model,
#         args.output_dir_aug,
#     )


print("\nSTEP (3): Augmenting Images via Prompt-Based Editing")
for i, dm_ in enumerate(diff_models_LANCE):
    print(f"\n     (3.{i+1}) Editing with {dm_}\n")
    for ds in list(file_type_map.keys()):
        subprocess.run(
            [
                "python",
                "-m",
                "SemiTruths.image_augmentation.prompt_based_editing.prompt_based_image_aug",
                "--dset_name",
                "ImageFolder",
                "--img_dir",
                args.input_data_pth,
                "--json_path",
                args.input_metadata_pth,
                "--ldm_type",
                dm_,
                "--lance_path",
                args.lance_pth,
                "--dataset",
                ds,
                "--editcap_dict_path",
                "./data/editcap_dict.json",
                "--gencap_dict_path",
                "./data/prompt-prompt/gencap_dict.json",
            ],
        )

pdb.set_trace()

print("\nSTEP (4): Computing Augmentation Metadata & Change Metrics")
subprocess.run(
    [
        "python",
        "change_metrics/change_metrics.py",
        "--root_csv",
        args.metadata,
        "--root_dir",
        args.input_data,
        "--save_dir",
        args.input_data,
    ],
)

print("\nSTEP (5): Computing Quality Metrics")
for i, dm_ in enumerate(diff_models):
    print(f"\n     (5.{i+1}) Quality Checking Images from {dm_}\n")
    for ds in list(file_type_map.keys()):
        subprocess.run(
            [
                "python",
                "quality_check/postgen_quality_check.py",
                "--metadata",
                args.metadata,
                "--input_data_parent",
                args.input_data,
                "--output_file",
                args.qual_output_path,
                "--pert_data_parent",
                args.output_dir_aug,
                "--dataset",
                ds,
                "--model",
                dm_,
            ],
        )

print("\nSemi-Truths pipeline is complete! Data saved at ./data/.")
