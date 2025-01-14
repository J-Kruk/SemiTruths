import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import pdb

PATH_TO_PERTURB_DATA_PARENT = ""
PATH_TO_DATA_PARENT_IMG = ""
EDIT_EXTENSION = ".png"
PATH_TO_RESULT_CSV = ''



DS_extension_formats ={
        "ADE20K": ".jpg",
        "CelebAHQ": ".jpg",
        "CityScapes": ".png",
        "HumanParsing": ".png",
        "OpenImages": ".jpg",
        "SUN_RGBD": ".jpg",
    }

#import all csv files in directory

all_files = glob("/postgen_qc/*.csv")
list_all_df = [pd.read_csv(f) for f in all_files]
combined_df = pd.concat(list_all_df)
combined_df["index"] = range(len(combined_df))
metric_name = ['img1_img2', 'cap2_img2', 'brisque_score_orig', 'brisque_score_perturb', 'direct_sim']

# Plot the metrics
axes_lim = [[0,1] , [0,1] , [0,100] , [0,100] , [0,1]]
for metric in metric_name:
    plt.figure()
    # Plot the metric
    plt.hist(combined_df[metric].dropna(), bins=100, color='blue')
    plt.xlim(axes_lim[metric_name.index(metric)])
    plt.xlabel('Index')
    plt.ylabel(metric)
    plt.title(f'{metric} vs Index')
    # Save the graph
    plt.savefig(f'metric_graph.png')

# On observing brisque score of perturbed images we see two distinct guassian distributions

# BRISQUE
selected_metric = 'brisque_score_perturb'
# data = combined_df[selected_metric].dropna().values.reshape(-1,1)
threshold = [0,70]
filtered = combined_df[combined_df['brisque_score_perturb'] >=threshold[0]]
print("Original_data ", len(filtered))
filtered = filtered[filtered['brisque_score_perturb'] <=threshold[1]]
print("Filtered data after brisque score ", len(filtered))

#IMG1_IMG2
selected_metric = 'img1_img2'
data = combined_df[selected_metric].dropna().values.reshape(-1,1)

threshold = [0.8816, 0.9896]
filtered = filtered[filtered[selected_metric] >= 0]
print("Original_data ", len(filtered))
filtered = filtered[filtered[selected_metric] >=threshold[0]]
filtered = filtered[filtered[selected_metric] <=threshold[1]]
print(f"Filtered data after {selected_metric} ", len(filtered))

selected_metric = 'direct_sim'
data = combined_df[selected_metric].dropna().values.reshape(-1,1)

threshold = [0.8115, 0.9786]
filtered = filtered[filtered[selected_metric] >= 0]
print("Original_data ", len(filtered))
filtered = filtered[filtered[selected_metric] >=threshold[0]]
filtered = filtered[filtered[selected_metric] <=threshold[1]]
print(f"Filtered data after {selected_metric} ", len(filtered))

combined_df['pass_qc'] = combined_df['index'].apply(lambda x: x in filtered['index'].values)
combined_df.to_csv(PATH_TO_RESULT_CSV, index=False)



