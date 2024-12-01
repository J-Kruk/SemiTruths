import argparse
from magnitude_change import *
import pdb
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description="Color Scheme for Skyscapes and Uavid")
parser.add_argument(
    "--root_csv",
    type=str,
    default="/srv/share4/apal72/half-truths/csvs/consolidated_meta_no_helen.csv",
    help="main csv",
)
parser.add_argument(
    "--root_dir",
    type=str,
    default="/srv/share4/apal72/half-truths/data/Half_Truths_Dataset",
    help="root directory to the images",
)
parser.add_argument(
    "--model", type=str, default="mistral_inpainting", help="model name"
)
parser.add_argument("--save_dir", type=str, default=None)

args = parser.parse_args()

datasets = []
models = []

data = pd.read_csv(args.root_csv)
data = data.sort_values(by="method", ascending=False)
file_type_map = {
    "ADE20K": ".jpg",
    "CelebAHQ": ".jpg",
    "CityScapes": ".png",
    "HumanParsing": ".png",
    "OpenImages": ".jpg",
    "SUN_RGBD": ".jpg",
}

change_calc = Change(save_dir=args.save_dir)
metrics = []

for i, row in tqdm(data.iterrows(), total=len(data)):
    real_path = os.path.join(args.root_dir, row["image_path"])
    if not os.path.exists(real_path):
        real_path = os.path.join(
            args.root_dir,
            "Half_Truths_Dataset",
            "images",
            row["dataset"],
            row["image_id"] + file_type_map[row["dataset"]],
        )
    fake_path = os.path.join(args.root_dir, row["perturbed_image_path"])
    perturbed_label = row["perturbed_mask_name"]
    og_label = row["mask_name"]

    try:
        metrics.append(
            change_calc.calc_metrics(real_path, fake_path, og_label, perturbed_label)
        )
    except Exception as error:
        print(f"ERROR:   {error}")
        pdb.set_trace()

columns = [
    "dreamsim",
    "lpips_score",
    "sen_sim",
    "clip_sim",
    "mse_rgb",
    "mse_gray",
    "ssim_rgb",
    "ssim_gray",
    "ratio_rgb",
    "ratio_gray",
    "largest_component_size_gray",
    "largest_component_size_rgb",
    "cc_cluters_rgb",
    "cc_clusters_gray",
    "cluster_dist_rgb",
    "cluster_dist_gray",
]
metrics_data = pd.DataFrame(metrics, columns=columns)
final_data = pd.concat([data, metrics_data], axis=1)

name = os.path.splitext(args.root_csv.split("/")[-1])[0] + "_mag-metrics.csv"
pdb.set_trace()

if not (os.path.exists(os.path.join(args.root_dir, "mag_csvs"))):
    os.makedirs(os.path.join(args.root_dir, "mag_csvs"))
final_data.to_csv(os.path.join(args.root_dir, "mag_csvs", name), index=False)
