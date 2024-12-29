import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import pdb


def apply_quality_thresholds(qc_files_path, output_path, metric_thresholds):
    """
    Applies pre-computed thresholds on quality metrics that
    determine which images are highly salient, and which are not.

    Inputs
    ---------
    qc_files_path : str (path)
        Path to directory containing quality metric metadata files.
    output_path : str (path)
        Path where consolidated metadata will be saved.
    metric_thresholds : dict
        Dictionary where keys are the names of quality metrics columns
        and the values contain the lower & upper value thresholds.
    """

    all_files = glob(f"{qc_files_path}/**/*.csv", recursive=True)
    list_all_df = [pd.read_csv(f) for f in all_files]
    combined_df = pd.concat(list_all_df)
    combined_df["index"] = range(len(combined_df))

    # BRISQUE
    selected_metric = "brisque_score_perturb"
    threshold = metric_thresholds[selected_metric]
    filtered = combined_df[combined_df["brisque_score_perturb"] >= threshold[0]]
    print("Original_data ", len(filtered))
    filtered = filtered[filtered["brisque_score_perturb"] <= threshold[1]]
    print("Filtered data after brisque score ", len(filtered))

    # CAP2-IMG2 SIM
    selected_metric = "cap2_img2"
    data = filtered[selected_metric].dropna().values.reshape(-1, 1)

    threshold = metric_thresholds[selected_metric]
    filtered = filtered[filtered[selected_metric] >= 0]
    print("Original_data ", len(filtered))
    filtered = filtered[filtered[selected_metric] >= threshold[0]]
    filtered = filtered[filtered[selected_metric] <= threshold[1]]
    print(f"Filtered data after {selected_metric} ", len(filtered))

    # IMG1-IMG2 SIM
    selected_metric = "img1_img2"
    data = combined_df[selected_metric].dropna().values.reshape(-1, 1)

    threshold = metric_thresholds[selected_metric]
    filtered = filtered[filtered[selected_metric] >= 0]
    print("Original_data ", len(filtered))
    filtered = filtered[filtered[selected_metric] >= threshold[0]]
    filtered = filtered[filtered[selected_metric] <= threshold[1]]
    print(f"Filtered data after {selected_metric} ", len(filtered))

    # DIRECTIONAL SIM
    selected_metric = "direct_sim"
    data = combined_df[selected_metric].dropna().values.reshape(-1, 1)

    threshold = metric_thresholds[selected_metric]
    filtered = filtered[filtered[selected_metric] >= 0]
    print("Original_data ", len(filtered))
    filtered = filtered[filtered[selected_metric] >= threshold[0]]
    filtered = filtered[filtered[selected_metric] <= threshold[1]]
    print(f"Filtered data after {selected_metric} ", len(filtered))

    combined_df["pass_qc"] = combined_df["index"].apply(
        lambda x: x in filtered["index"].values
    )

    combined_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    PATH_TO_QC_METRICS = "./data/gen/qc_meta_files"
    PATH_TO_OUTPUT_CSV = "./data/gen/semitruths_metadata.csv"

    metric_thresholds = {
        "img1_img2": [0.8816, 0.9896],
        "cap2_img2": [0.2083, 0.2971],
        "brisque_score_perturb": [0, 70],
        "direct_sim": [0.8115, 0.9786],
    }

    apply_quality_thresholds(PATH_TO_QC_METRICS, PATH_TO_OUTPUT_CSV, metric_thresholds)
