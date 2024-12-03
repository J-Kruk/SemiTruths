import os
import torch
from PIL import Image
from utils import *
from tqdm import tqdm
import pandas as pd
import pdb
import sys
torch.set_default_tensor_type("torch.cuda.FloatTensor")


def post_qual_check(
    img_dataframe,
    DS,
    PATH_TO_DATA_PARENT,
    PATH_TO_PERTURB_DATA_PARENT,
    DS_EXTENSION,
    EDIT_EXTENSION,
    CSV_POSTGEN_QC,
):

    for i, row in tqdm(img_dataframe.iterrows(), total=len(img_dataframe)):
        # if type(row["cap2_img2"]) == float:
        model = row["model"]

        try:
            orig_img_path = os.path.join(
                PATH_TO_DATA_PARENT, DS, row["image_id"] + DS_EXTENSION
            )
        except KeyError:
            orig_img_path = os.path.join(
                PATH_TO_DATA_PARENT, DS, row["img_id"] + DS_EXTENSION
            )
        orig_img = Image.open(orig_img_path)

        orig_caption = row["mask_name"]
        if len(orig_img.getbands()) != 3:
            print(f"Image shape error for:   {orig_img_path}")
            continue

        # if perturbed image path does not exist, return:
        if not os.path.exists(
            os.path.join(
                PATH_TO_PERTURB_DATA_PARENT,
                row["dataset"],
                model,
                row["mask_id"] + "_" + DS + "_" + model + EDIT_EXTENSION,
            )
        ):
            print("perturbed image does not exist")
            print(
                os.path.join(
                    PATH_TO_PERTURB_DATA_PARENT,
                    row["dataset"],
                    model,
                    row["mask_id"] + "_" + DS + "_" + model + EDIT_EXTENSION,
                )
            )

        perturbed_img_path = os.path.join(
            PATH_TO_PERTURB_DATA_PARENT,
            row["dataset"],
            model,
            row["mask_id"] + "_" + DS + "_" + model + EDIT_EXTENSION,
        )

        perturbed_img = Image.open(perturbed_img_path)
        perturbed_caption = row["perturbed_label"]

        cap2_img2 = calculate_image_caption_clip_similarity(
            perturbed_img, perturbed_caption
        )
        direct_sim = calculate_directional_similarity(
            orig_img, orig_caption, perturbed_img, perturbed_caption
        )
        img1_img2 = calculate_image_similarity(orig_img, perturbed_img)
        brisque_score_perturb = brisque_Score(perturbed_img)
        brisque_score_orig = brisque_Score(orig_img)

        img_dataframe.at[i, "cap2_img2"] = cap2_img2
        img_dataframe.at[i, "direct_sim"] = direct_sim
        img_dataframe.at[i, "img1_img2"] = img1_img2
        img_dataframe.at[i, "brisque_score_orig"] = brisque_score_orig
        img_dataframe.at[i, "brisque_score_perturb"] = brisque_score_perturb

        if i % 25 == 0:
            img_dataframe.to_csv(CSV_POSTGEN_QC, index=False)

        img_dataframe.to_csv(CSV_POSTGEN_QC, index=False)


def postgen_quality_check(
    CSV_READ_FILE,
    CSV_POSTGEN_QC,
    DS,
    PATH_TO_DATA_PARENT,
    PATH_TO_PERTURB_DATA_PARENT,
    DS_EXTENSION,
    EDIT_EXTENSION,
):

    if os.path.exists(CSV_POSTGEN_QC):
        gen_image_df = pd.read_csv(CSV_POSTGEN_QC)
    else:
        gen_image_df = pd.read_csv(CSV_READ_FILE)
        quality_columns = [
            "cap2_img2",
            "direct_sim",
            "img1_img2",
            "brisque_score_orig",
            "brisque_score_perturb",
        ]
        for q_c in quality_columns:
            gen_image_df[q_c] = pd.NA

    post_qual_check(
        gen_image_df,
        DS,
        PATH_TO_DATA_PARENT,
        PATH_TO_PERTURB_DATA_PARENT,
        DS_EXTENSION,
        EDIT_EXTENSION,
        CSV_POSTGEN_QC,
    )


if __name__ == "__main__":
    PATH_TO_DATA_PARENT = "/srv/hoffman-lab/share4/apal72/half-truths/data/Half_Truths_Dataset/images"
    CSV_POSTGEN_QC_ = (
        "/srv/hoffman-lab/flash9/apal72/half-truths/Semi-Truths/metadata/qc"
    )
    PATH_TO_PERTURB_DATA_PARENT = "/srv/hoffman-lab/flash9/apal72/half-truths/Semi-Truths/inpainting"
    EDIT_EXTENSION = ".png"

    DATASETS = [
        sys.argv[1]
    ]
    MODELS = [
        sys.argv[2] #
        # "StableDiffusion_v4",
        # "StableDiffusion_XL",
        # "StableDiffusion_v5",  #
        # "OpenJourney",
    ]

    DS_extemsion_dict = {
        "ADE20K": ".jpg",
        "CelebAHQ": ".jpg",
        "CityScapes": ".png",
        "HumanParsing": ".png",
        "OpenImages": ".jpg",
        "SUN_RGBD": ".jpg",
    }

    for DS in DATASETS:
        for MODEL in MODELS:
            CSV_READ_FILE = f"{PATH_TO_PERTURB_DATA_PARENT}/{DS}/{MODEL}/{DS}_{MODEL}_meta.csv"
            CSV_POSTGEN_QC = f"{CSV_POSTGEN_QC_}/{DS}/{DS}_{MODEL}_meta_qc_size.csv"
            if not os.path.exists(f"{CSV_POSTGEN_QC_}/{DS}"):
                os.mkdir(f"{CSV_POSTGEN_QC_}/{DS}")
                
            postgen_quality_check(
                CSV_READ_FILE,
                CSV_POSTGEN_QC,
                DS,
                PATH_TO_DATA_PARENT,
                PATH_TO_PERTURB_DATA_PARENT,
                DS_extemsion_dict[DS],
                EDIT_EXTENSION,
            )
