import sys
from tqdm.notebook import tqdm
from diffusers import (
    StableDiffusionInpaintPipeline,
    AutoPipelineForInpainting,
    DiffusionPipeline,
    IFInpaintingSuperResolutionPipeline,
)
from datasets import load_dataset
import pandas as pd
import PIL
from PIL import Image, ImageFilter
import os
import json
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch
import ipdb
from tqdm import tqdm
from huggingface_hub import login
import argparse

# hugging face token for reading our input dataset:
login(token="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")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--diff_model",
    default="StableDiffusion_v4",
    help="Name of diffusion algorithm for augmentation.",
)
parser.add_argument(
    "--input_data",
    default="../../data/input",
    help="Path to input media.",
)
parser.add_argument(
    "--output_dir",
    default="../../data/gen",
    help="Path to augmented media.",
)
parser.add_argument(
    "--pert_file",
    default="../../data/metadata_pert.json",
    help="Path to json containing label perturbations.",
)
args = parser.parse_args()


# MODEL DICT --> { <model name> : <model path hf> }
diff_model_dict = {
    "StableDiffusion_v4": "CompVis/stable-diffusion-v1-4",
    "StableDiffusion_v5": "runwayml/stable-diffusion-v1-5",
    "StableDiffusion_XL": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "Kandinsky_2_2": "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    "OpenJourney": "prompthero/openjourney",
}

DATASETS = [
    "HumanParsing",
    "CelebAHQ",
    "SUN_RGBD",
    "ADE20K",
    "CityScapes",
    "OpenImages",
]
# DATASET = "ADE20K"
# DIFF_MODEL = "StableDiffusion_v4"

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
if not os.path.exists(os.path.join(args.output_dir, "inpainting")):
    os.makedirs(os.path.join(args.output_dir, "inpainting"))

for ds in DATASETS:
    dir_ = os.path.join(args.output_dir, "inpainting", ds, args.diff_model)
    if not os.path.exists(os.path.join(args.output_dir, "inpainting", ds)):
        os.mkdir(os.path.join(args.output_dir, "inpainting", ds))
    if not os.path.exists(dir_):
        os.mkdir(dir_)

# PATH DECLARATIONS
# IMG_DIR = "/data/jkruk3/half-truths/mistral_inpainting"  # "... output dir for generated images ..."
# OUT_DIR = os.path.join(IMG_DIR, DATASET, args.diff_model)
# PERT_LABEL_DIR = "/data/jkruk3/half-truths/mistral_7b"  # "... dir containing metadata with mask label perturbations ..."

# --------------------------------------------------------------------------------- #

# loading input dataset from huggingface:
input_data = load_dataset("Half-Truths-Project/base-datasets-3")

# merging all dataset metadata into one DF:
perturbed_labels = {}
for file in os.listdir(args.pert_file):
    with open(os.path.join(args.pert_file, file), "r") as f:
        data_ = json.load(f)
        ds = data_[list(data_.keys())[0]]["dataset"]
        perturbed_labels[ds] = data_
print(f"Benchmarks in input data:  {list(perturbed_labels.keys())}")

# processing input data --> removing data flagged for poor quality:
try:
    input_data_ds = pd.DataFrame(input_data[DATASET])
except KeyError:
    # input_data_ds = pd.DataFrame(input_data[15000:25000]) #CelebA
    input_data_ds = pd.DataFrame(input_data[30000:40000])  # ADE20k
    print(f"KeyError while loading data, input length ::  {len(input_data_ds)}")
input_data_ds = input_data_ds.loc[
    (input_data_ds.ratio != "1000.0") | (input_data_ds.mask_name != "NA")
]

pert_labels = []
sem_changes = []
quality_flags = []
for i, row in tqdm(input_data_ds.iterrows(), total=len(input_data_ds)):
    try:
        meta = perturbed_labels[DATASET][row.img_id]
    except KeyError:
        # this logic is exclusively for CelebAHQ dataset:
        meta = perturbed_labels[DATASET][f"CelebA_{row.img_id}"]

    try:
        if DATASET == "CelebAHQ":
            mask_meta = [
                m
                for m in meta["objects"]
                if m["mask_path"][
                    m["mask_path"].rfind("/") + 1 : m["mask_path"].rfind(".")
                ]
                == row.mask_id
            ][0]
        else:
            mask_meta = [m for m in meta["objects"] if m["id"] == row.mask_id][0]
        pert_labels.append(mask_meta["target"])
        sem_changes.append(mask_meta["sem_magnitude"])
        quality_flags.append(mask_meta["target_qual_flag"])

    except IndexError:
        pert_labels.append(pd.NA)
        sem_changes.append(pd.NA)
        quality_flags.append(pd.NA)

input_data_ds["perturbed_label"] = pert_labels
input_data_ds["sem_magnitude"] = sem_changes
input_data_ds["quality_flag"] = quality_flags

input_data_ds_qual = input_data_ds[input_data_ds.quality_flag == False]
input_data_ds_qual = input_data_ds_qual.dropna(subset=["perturbed_label"])
input_data_ds_qual = input_data_ds_qual.reset_index(drop=True)

print(f"Number of masks w/ perturbed labels in total ::  {len(input_data_ds)}")
print(f"Number of masks w/ quality flag NOT raised ::  {len(input_data_ds_qual)}")
print(
    f"Percent of `quality` mask label perturbations ::  {len(input_data_ds_qual) / len(input_data_ds) * 100}"
)

# preparing output metadata file --> `<dataset>_<model>_meta.csv`
META_FILE = os.path.join(
    args.output_dir, DATASET, f"{DATASET}_{args.diff_model}_meta.csv"
)
if os.path.exists(META_FILE):
    aug_meta = pd.read_csv(META_FILE)
else:
    aug_meta_columns = input_data_ds_qual.columns.tolist() + [
        "dataset",
        "method",
        "model",
        "perturbed_path",
    ]
    aug_meta = pd.DataFrame(columns=aug_meta_columns)


# INPAINTING HELPER FUNCTIONS --
def inpaint_img(pipe, row, mask_blur=16):
    """
    Create an inpainted image using the org_image,
    mask, and perturbed mask label.
    """

    prompt = row["perturbed_label"]
    org_img = row["image"].convert("RGB")
    mask = row["mask"].convert("L")

    # mask prep to improve inpainting:
    mask = mask.filter(ImageFilter.MaxFilter(size=9))
    mask = mask.filter(ImageFilter.MaxFilter(size=9))
    mask = mask.filter(ImageFilter.ModeFilter(size=10))
    try:
        mask = pipe.mask_processor.blur(mask, blur_factor=mask_blur)
    except:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))

    # inpainting:
    try:
        inpainted = pipe(
            prompt=prompt, image=org_img, mask_image=mask, device_map="auto"
        ).images[0]
    except:
        inpainted = pipe(prompt=prompt, image=org_img, mask_image=mask).images[0]
    return inpainted


def create_img_augmentations(
    input_data,
    meta,
    diff_model_dict,
    model_name=args.diff_model,
    save_img_dir=OUT_DIR,
    save_meta_file=META_FILE,
):
    """
    This function will generate perturbed images for each
    image in the input metadata. There will exist one perturbed image
    for each inpainting model listed in the diff_model_dict.

    Returns a new metadata file with information on perturbed images
    in obj['generations'] for each image.
    """

    print(f"\nGenerating Inpainted Images with ::   {model_name}\n")
    model_path = diff_model_dict[model_name]

    if "Floyd" in model_name:
        pipe = DiffusionPipeline.from_pretrained(
            model_path, variant="fp16", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        pipe = pipe.to("cuda")
    else:
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_path, torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        pipe = pipe.to("cuda")

    for i, row in tqdm(input_data.iterrows(), total=len(input_data)):

        inpaint_path = os.path.join(
            save_img_dir, f"{row['mask_id']}_{DATASET}_{model_name}.png"
        )
        if not os.path.exists(inpaint_path):
            inpainted_img = inpaint_img(pipe, row)
            inpainted_img.save(inpaint_path)
        else:
            print(f"Image already exists :: {inpaint_path}")

        row["dataset"] = DATASET
        row["method"] = "inpainting"
        row["model"] = model_name
        row["perturbed_path"] = inpaint_path
        if len(meta) == 0:
            meta = pd.DataFrame(row).T
        else:
            meta = pd.concat([meta, pd.DataFrame(row).T], ignore_index=True)

        if len(meta) % 10 == 0:
            meta.to_csv(save_meta_file, index=False)

    meta.to_csv(save_meta_file, index=False)
    return meta


# RUN INPAINTING --
gen_meta = create_img_augmentations(input_data_ds_qual, aug_meta, diff_model_dict)
print("DONE!!!!")
