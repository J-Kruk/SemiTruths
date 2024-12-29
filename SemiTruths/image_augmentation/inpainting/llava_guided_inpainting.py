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
import pdb
from tqdm import tqdm
from huggingface_hub import login
import argparse

# MODEL DICT --> { <model name> : <model path hf> }
diff_model_dict = {
    "StableDiffusion_v4": "CompVis/stable-diffusion-v1-4",
    "StableDiffusion_v5": "runwayml/stable-diffusion-v1-5",
    "StableDiffusion_XL": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "Kandinsky_2_2": "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    "OpenJourney": "prompthero/openjourney",
}

# INPUT DATASETS (Semantic Segmentation)
DATASETS = [
    "HumanParsing",
    "CelebAHQ",
    "SUN_RGBD",
    "ADE20K",
    "CityScapes",
    "OpenImages",
]

# NOTE: hugging face token for reading our input dataset:
# login(token="")

# loading input dataset from huggingface:
# input_data = load_dataset("Half-Truths-Project/base-datasets-3")


def prepare_directory_struct(diff_model, output_dir_img_aug):
    """
    Builds a file structure for perturbed / generated
    labels and images. Works from directories declared
    in args.

    Inputs:
    ----------------
    diff_model : str
        Name of diffusion model used in inpainting.
    output_dir_img_aug : str (Path)
        Path where to save inpainted images and perturned
        captions.
    """

    if not os.path.exists(output_dir_img_aug):
        os.makedirs(output_dir_img_aug)
    if not os.path.exists(os.path.join(output_dir_img_aug, "inpainting")):
        os.makedirs(os.path.join(output_dir_img_aug, "inpainting"))

    for ds in DATASETS:
        dir_ = os.path.join(output_dir_img_aug, "inpainting", ds, diff_model)
        if not os.path.exists(os.path.join(output_dir_img_aug, "inpainting", ds)):
            os.mkdir(os.path.join(output_dir_img_aug, "inpainting", ds))
        if not os.path.exists(dir_):
            os.mkdir(dir_)


def filter_labels(pert_file, input_meta):
    """
    Filters out low-quality perturbed mask labels before
    using them for conditional image perturbation.

    Inputs:
    ----------------
    pert_file : str (Path)
        Path to json file with perturbed mask labels.
    input_meta : str (Path)
        Path to csv file with input, real image metadata.

    Returns:
    ----------------
    input_data_ds_qual : pd.DataFrame
        DataFrame of input, real images filtered on perturbed
        label quality, ready for inpainting.
    """

    perturbed_labels = {}
    with open(pert_file, "r") as f:
        data_ = json.load(f)
        perturbed_labels = data_

    input_data_ds = pd.read_csv(input_meta)
    input_data_ds = input_data_ds.loc[
        (input_data_ds.ratio != "1000.0") | (input_data_ds.mask_name != "NA")
    ]

    pert_labels = []
    sem_changes = []
    quality_flags = []
    for i, row in tqdm(input_data_ds.iterrows(), total=len(input_data_ds)):
        try:
            meta = perturbed_labels[row.image_id]
        except KeyError:
            # this logic is exclusively for CelebAHQ dataset:
            meta = perturbed_labels[f"CelebA_{row.image_id}"]

        try:
            if row.dataset == "CelebAHQ":
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

    return input_data_ds_qual


# INPAINTING HELPER FUNCTIONS --
def inpaint_img(pipe, row, input_data_pth, mask_blur=16):
    """
    Create an inpainted image using the org_image,
    mask, and perturbed mask label. Preprocesses the image
    mask before augmentation.

    Inputs:
    ----------------
    pipe : HuggingFace Pipeline
        Diffusion conditional inpainting model pipeline.
    row : pd.Series
        Row of metadata with image information.
    input_data_pth : str (Path)
        Path to input, real images.
    mask_blur : int
        Blur factor for mask Gaussian blur.

    Returns:
    ----------------
    inpainted : PNG
        Diffusion inpainted image.
    """

    prompt = row["perturbed_label"]
    try:
        org_img = row["image"].convert("RGB")
        mask = row["mask"].convert("L")
    except KeyError:
        org_img = Image.open(os.path.join(input_data_pth, row["image_path"])).convert(
            "RGB"
        )
        mask = Image.open(os.path.join(input_data_pth, row["mask_path"])).convert("L")

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
    input_data_ds_qual,
    input_data_pth,
    diff_model_dict,
    model_name,
    save_img_dir,
):
    """
    This function will generate perturbed images for each
    image in the input metadata. There will exist one perturbed image
    for each inpainting model listed in the diff_model_dict.

    Saves a new metadata file with information on perturbed images
    in obj['generations'] for each image.

    Inputs:
    ----------------
    input_data_ds_qual : pd.DataFrame
        DataFrame containing input data filered on quality of perturbed
        image mask labels.
    input_data_pth : str (Path)
        Path to input, real image data.
    diff_model_dict : dict
        Keys contain diffusion model name, values contain huggingface paths.
    model_name : str
        Name of diffusion model to use.
    save_img_dir : str (Path)
        Path to save inpainted images.
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

    for ds in input_data_ds_qual.dataset.unique():
        ds_input_data_qual = input_data_ds_qual.loc[input_data_ds_qual.dataset == ds]
        save_meta_file = os.path.join(
            save_img_dir,
            "inpainting",
            ds,
            model_name,
            f"{ds}_{model_name}_meta.csv",
        )

        if os.path.exists(save_meta_file):
            aug_meta = pd.read_csv(save_meta_file)
        else:
            aug_meta_columns = input_data_ds_qual.columns.tolist() + [
                "dataset",
                "method",
                "model",
                "perturbed_path",
            ]
            aug_meta = pd.DataFrame(columns=aug_meta_columns)

        for i, row in tqdm(
            ds_input_data_qual.iterrows(), total=len(ds_input_data_qual)
        ):
            inpaint_path = os.path.join(
                save_img_dir,
                "inpainting",
                row.dataset,
                model_name,
                f"{row['mask_id']}_{row.dataset}_{model_name}.png",
            )
            if not os.path.exists(inpaint_path):
                inpainted_img = inpaint_img(pipe, row, input_data_pth)
                inpainted_img.save(inpaint_path)
            else:
                print(f"Image already exists :: {inpaint_path}")

            row["dataset"] = row.dataset
            row["method"] = "inpainting"
            row["model"] = model_name
            row["perturbed_path"] = inpaint_path
            if len(aug_meta) == 0:
                aug_meta = pd.DataFrame(row).T
            else:
                aug_meta = pd.concat([aug_meta, pd.DataFrame(row).T], ignore_index=True)

            if len(aug_meta) % 10 == 0:
                aug_meta.to_csv(save_meta_file, index=False)

        aug_meta = aug_meta.drop_duplicates()
        aug_meta.to_csv(save_meta_file, index=False)


if __name__ == "__main__":
    ## ARGPARSE ##
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--diff_model",
        default="StableDiffusion_v4",
        help="Name of diffusion algorithm for augmentation.",
    )
    parser.add_argument(
        "--input_data_pth",
        default="../../data/input",
        help="Path to input media.",
    )
    parser.add_argument(
        "--input_meta_pth",
        default="../../data/input/metadata.csv",
        help="Path to input data csv.",
    )
    parser.add_argument(
        "--output_dir_img_aug",
        default="../../data/gen",
        help="Path to augmented media.",
    )
    parser.add_argument(
        "--pert_file",
        default="../../data/metadata_pert.json",
        help="Path to json containing label perturbations.",
    )
    args = parser.parse_args()

    ## RUN INPAINTING ##
    prepare_directory_struct(args.diff_model, args.output_dir_img_aug)
    input_data_ds_qual = filter_labels(args.pert_file, args.input_meta)

    create_img_augmentations(
        input_data_ds_qual,
        args.input_data_pth,
        diff_model_dict,
        args.diff_model,
        args.output_dir_img_aug,
    )
