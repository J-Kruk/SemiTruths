import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pdb 
import ast
import numpy as np
import argparse 

parser = argparse.ArgumentParser(description="Color Scheme for Skyscapes and Uavid")
parser.add_argument('--generate',  action='store_true',help='root directory to the labels, ClearNoon') 
parser.add_argument('--root_csv', type=str, default='/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/metadata_flat.csv',help='root directory to the labels, ClearNoon')  
args = parser.parse_args()

root_csv = args.root_csv
df = pd.read_csv(root_csv)
# df = df_.head(100)
entities = df['entities']
img_id = df['image_id']
dataset = df['dataset'].unique()
scene_diversity = []
scene_complexity = []

def plot(values,small_max, medium_max,text):
    plt.figure()
    plt.hist(values, bins=100, edgecolor='black')

    # Add title and labels
    plt.title(text)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.axvline(x=small_max, color='r', linestyle='--', label='Small/Medium cutoff')
    plt.axvline(x=medium_max, color='g', linestyle='--', label='Medium/Large cutoff')
    plt.legend()
    # Save the plot to a file
    plt.savefig('plots/' + text+'.png')
    plt.close()

def bin_values(values, values_1):
    scene = []
    mean = np.mean(values_1)
    std_dev = np.std(values_1)
    # Define the bin boundaries
    small_max = mean - std_dev
    medium_max = mean + std_dev
    for i in values:
        if(i < small_max):
            scene.append('low')
        elif(small_max <= i < medium_max):
            scene.append('medium')
        else:
            scene.append('high')
    
    return scene, small_max, medium_max


if(args.generate):
    for i, entity in tqdm(enumerate(entities),total=len(entities)):
        scene_diversity.append(len(ast.literal_eval(entity)))
        count = df['image_id'].value_counts().get(img_id[i], 0)
        scene_complexity.append(count)
    df['scene_diversity'] = scene_diversity
    df['scene_complexity'] = scene_complexity
else:
    scene_diversity = df['scene_diversity']
    scene_complexity = df['scene_complexity']
scene_diversity_1 = []
scene_complexity_1 = []
seen = []
# pdb.set_trace()
for i in tqdm(range(len(img_id)),total=len(img_id)):
    if img_id[i] in seen:
        continue
    else:
        seen.append(img_id[i])
        scene_diversity_1.append(scene_diversity[i])
        scene_complexity_1.append(scene_complexity[i])

# pdb.set_trace()
df['scene_diversity_bins'], small_max_diverse, medium_max_diverse = bin_values(scene_diversity, scene_diversity_1)
df['scene_complexity_bins'], small_max_complex, medium_max_complex = bin_values(scene_complexity, scene_complexity_1)

if(args.generate):
    df.to_csv(root_csv.split('.')[0] + '_scene_diversity_complexity.csv')
plot(scene_diversity_1, small_max_diverse, medium_max_diverse,'Data_Scene_Diversity')
plot(scene_complexity_1, small_max_complex, medium_max_complex,'Data_Scene_Complexity')


# pdb.set_trace()
for d in dataset:
    filtered_df = df[df['dataset'] == d]
    filtered_df_1 = filtered_df.drop_duplicates(subset='image_id')
    scene_diversity_filtered = filtered_df_1['scene_diversity']
    scene_complexity_filtered = filtered_df_1['scene_complexity']
    plot(scene_diversity_filtered, small_max_diverse, medium_max_diverse, (d +'_Scene_Diversity'))
    plot(scene_complexity_filtered, small_max_complex, medium_max_complex, (d +'_Scene_Complexity'))






