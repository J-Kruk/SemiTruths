"""
This script will run each component in the pipeline to create the Semi-Truths
dataset from a sample of processed semantic segmentation data.

---------------------------------

Copyright (c) 2024 <Authors>

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
import subprocess
import pandas as pd
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--input_data",
    default="../../data/input",
    help="Path to input media.",
)
parser.add_argument(
    "--metadata",
    default="../../data/input/metadata.csv",
    help="Path to input metadata.",
)
parser.add_argument(
    "--llava_model",
    default="liuhaotian/llava-v1.6-mistral-7b",
    help="LLaVA-Mistral huggingface model id.",
)
parser.add_argument(
    "--llava_cache_dir",
    default="../llava_cache",
    help="Directory to store LLaVA cache files.",
)
parser.add_argument(
    "--output_dir_pert",
    default="../../data",
    help="Directory to save metadata with perturbed mask labels.",
)
parser.add_argument(
    "--output_dir_aug",
    default="../../data/gen",
    help="Directory to save metadata with perturbed mask labels.",
)
parser.add_argument(
    "--pert_file",
    default="../../data/metadata_pert.json",
    help="Path to json containing label perturbations.",
)
args = parser.parse_args()

diff_models = [
    "StableDiffusion_v4",
    "StableDiffusion_v5",
    "StableDiffusion_XL",
    "Kandinsky_2_2",
    "OpenJourney",
]


print("STEP (1): Perturbing Mask Labels w/ LLaVA-Mistral-7b for Inpainting")
subprocess.run(
    [
        "python",
        "image_augmentation/inpainting/llava_mask_label_pert.py",
        "--input_data",
        args.input_data,
        "--metadata",
        args.metadata,
        "--llava_model",
        args.llava_model,
        "--llava_cache_dir",
        args.llava_cache_dir,
        "--output_dir",
        args.output_dir_pert,
    ],
    shell=True,
    check=True,
)

print("\nSTEP (2): Augmenting Images via Diffusion Inpainting")
for i, dm_ in enumerate(diff_models):
    print(f"\n     (2.{i+1}) Inpainting with {dm_}\n")
    subprocess.run(
        [
            "python",
            "image_augmentation/inpainting/llava_guided_inpaiting.py",
            "--diff_model",
            dm_,
            "--input_data",
            args.input_data,
            "--output_dir",
            args.output_dir_aug,
            "--pert_file",
            args.pert_file,
        ],
        shell=True,
        check=True,
    )

print("\nSTEP (3): Computing Augmentation Metadata & Change Metrics")

print("\nSTEP (4): Computing Quality Metrics")
