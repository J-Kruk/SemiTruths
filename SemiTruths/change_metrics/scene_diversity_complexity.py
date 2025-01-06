import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pdb 
import ast
import numpy as np
import argparse 

'''
This script generates the scene diversity and scene complexity metrics for the images in the dataset
Scene diversity: Number of entities in the image
Scene complexity: Number of times the image appears in the dataset
The script generates the metrics and bins them into low, medium and high
The script also plots the histograms of the metrics
'''

parser = argparse.ArgumentParser(description="Scene Diversity and Complexity Bins")
parser.add_argument('--generate',  action='store_true',help='if the metrics have not been generated, set this flag to generate them') 
parser.add_argument('--root_csv', type=str, required=True,help='root csv for the real images')  
args = parser.parse_args()

# Load the data
root_csv = args.root_csv
df = pd.read_csv(root_csv)
entities = df['entities'] # List of entities in the image
img_id = df['image_id'] # Image ID
dataset = df['dataset'].unique() # Dataset
scene_diversity = []
scene_complexity = []

def plot(values,small_max, medium_max,text):
    '''
    Function to plot the histogram of the values
    Inputs: values: list of values
            small_max: cutoff for small/medium
            medium_max: cutoff for medium/large
            text: title of the plot
    Outputs: None
    '''

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

    '''
    Function to bin the values
    Inputs: values: list of values
            values_1: list of values without duplicates
    Outputs: scene: list of bins
             small_max: cutoff for small/medium
             medium_max: cutoff for medium/large
    '''
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
        scene_diversity.append(len(ast.literal_eval(entity))) # Number of entities in the image
        count = df['image_id'].value_counts().get(img_id[i], 0) # Number of times the image appears in the dataset
        scene_complexity.append(count) 
    df['scene_diversity'] = scene_diversity
    df['scene_complexity'] = scene_complexity
else:
    scene_diversity = df['scene_diversity']
    scene_complexity = df['scene_complexity']
scene_diversity_1 = []
scene_complexity_1 = []
seen = []

# Remove duplicates
for i in tqdm(range(len(img_id)),total=len(img_id)):
    if img_id[i] in seen:
        continue
    else:
        seen.append(img_id[i])
        scene_diversity_1.append(scene_diversity[i])
        scene_complexity_1.append(scene_complexity[i])

# Bin the values
df['scene_diversity_bins'], small_max_diverse, medium_max_diverse = bin_values(scene_diversity, scene_diversity_1)
df['scene_complexity_bins'], small_max_complex, medium_max_complex = bin_values(scene_complexity, scene_complexity_1)

# Save the data to a csv file
if(args.generate):
    df.to_csv(root_csv.split('.')[0] + '_scene_diversity_complexity.csv')

# Plot the histograms
plot(scene_diversity_1, small_max_diverse, medium_max_diverse,'Data_Scene_Diversity')
plot(scene_complexity_1, small_max_complex, medium_max_complex,'Data_Scene_Complexity')


# Plot the histograms for each dataset
for d in dataset:
    filtered_df = df[df['dataset'] == d]
    filtered_df_1 = filtered_df.drop_duplicates(subset='image_id')
    scene_diversity_filtered = filtered_df_1['scene_diversity']
    scene_complexity_filtered = filtered_df_1['scene_complexity']
    plot(scene_diversity_filtered, small_max_diverse, medium_max_diverse, (d +'_Scene_Diversity'))
    plot(scene_complexity_filtered, small_max_complex, medium_max_complex, (d +'_Scene_Complexity'))





