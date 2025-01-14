import torch
from PIL import Image
from utils import *
from tqdm import tqdm
import pdb
from tqdm import tqdm
import csv
import os
import pandas as pd

torch.set_default_tensor_type("torch.cuda.FloatTensor")

CSV_READ_FILE = ""  # metadata for generated images

column_map = {
    "img_id": 0,
    "perturbed_img_id": 1,
    "original_caption": 2,
    "perturbed_caption": 3,
    "dataset": 4,
    "diffusion_model": 5,
    "language_model": 6,
    "sem_magnitude": 7,
}

formats = {
    "ADE20K": ".jpg",
    "CelebAHQ": ".jpg",
    "CityScapes": ".png",
    "HumanParsing": ".png",
    "OpenImages": ".jpg",
    "SUN_RGBD": ".jpg",
}

model_map = {
    "OJ": "OpenJourney",
    "SDv4": "StableDiffusion_v4",
    "SDv5": "StableDiffusion_v5",
}

DS = "SUN_RGBD"
MODEL = "SDv5"
LANG = "LlaVA-Hermes"

PATH_TO_ORIG_DATA_PARENT = ""  # path to original images
CSV_POSTGEN_QC = ""  # path to save postgen quality check results
PATH_TO_PERTURB_DATA_PARENT = ""  # path to perturbed images
EDIT_EXTENSION = ".jpeg"
DS_EXTENSION = formats[DS]


def post_qual_check_row(row, writer):
    """
    Calculate metrics for each row in the csv file
    Input: 
    row - list of values in the row
    writer - csv writer object
    
    Output:
    None. Function will write calculated values in the csv file
    
    """
    if len(row[column_map["img_id"]]):

        # remove model from orig path when just running independently
        # orig_img_path = os.path.join(PATH_TO_DATA_PARENT, DS,model, row[column_map['img_id']]+DS_EXTENSION)
        try:
            # print("Calculating metrics for ", row[column_map['img_id']])
            dataset = row[column_map["dataset"]]
            model = row[column_map["diffusion_model"]]
            language_model = row[column_map["language_model"]]
            if (dataset == DS) and (model == MODEL) and (language_model == LANG):
                print("Calculating metrics for ", row[column_map["perturbed_img_id"]])
                orig_img_path = os.path.join(
                    PATH_TO_ORIG_DATA_PARENT,
                    dataset,
                    row[column_map["img_id"]] + DS_EXTENSION,
                )
                orig_img = Image.open(orig_img_path)
                orig_caption = row[column_map["original_caption"]]
                brisque_score_orig = brisque_Score(orig_img)
                # if perturbed image path does not exist, return:

                perturbed_img_path = os.path.join(
                    PATH_TO_PERTURB_DATA_PARENT,
                    language_model,
                    dataset,
                    model_map[model],
                    row[column_map["perturbed_img_id"]],
                )
                # print(os.path.exists(perturbed_img_path))
                # print(perturbed_img_path)
                if not os.path.exists(perturbed_img_path):
                    return

                perturbed_img = Image.open(perturbed_img_path)

                # if not perturbed_img.getbbox():
                #     return

                perturbed_caption = row[column_map["perturbed_caption"]]

                cap2_img2 = calculate_image_caption_similarity(
                    perturbed_img, perturbed_caption
                )
                direct_sim = calculate_directional_similarity(
                    orig_img, orig_caption, perturbed_img, perturbed_caption
                )
                img1_img2 = calculate_image_similarity(orig_img, perturbed_img)
                brisque_score = brisque_Score(perturbed_img)

                # print(cap2_img2, direct_sim, img1_img2, brisque_score_orig, brisque_score)
                row.extend(
                    [
                        cap2_img2,
                        direct_sim,
                        img1_img2,
                        brisque_score_orig,
                        brisque_score,
                    ]
                )

                writer.writerow(row)
                print("Metrics calculated for ", row[column_map["perturbed_img_id"]])
        except Exception as e:
            print("Error in row: ", row)
            print(e)
        return


def postgen_quality_check():
    f = open(CSV_READ_FILE, "r")
    file = csv.reader(f)
    header = next(file)

    header_qc = header
    header_qc.extend(
        [
            "cap2_img2",
            "direct_sim",
            "img1_img2",
            "brisque_score_orig",
            "brisque_score_perturb",
        ]
    )

    out_f = open(CSV_POSTGEN_QC, "w")
    writer = csv.writer(out_f)
    writer.writerow(header_qc)

    i = 0
    for row in tqdm(file):
        post_qual_check_row(row, writer)

    f.close()
    out_f.close()


def main():
    postgen_quality_check()


if __name__ == "__main__":
    main()
