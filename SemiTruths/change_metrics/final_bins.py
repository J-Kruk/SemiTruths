import pandas as pd
import json
import math
from collections import Counter
import os
import sys 
import pdb
from tqdm import tqdm 
import argparse

def localized_diffused(df, dic):
    '''
    This function takes in the dataframe and the dictionary containing the values for the columns
    and returns the dataframe with the localization column added
    Input: df: dataframe
              dic: dictionary containing the values for the columns
    Output: df: dataframe with the localization column added, localized or diffused
    '''
    

    largest_comp_size = list(df['largest_component_size'])
    cluster_dist = list(df['cluster_dist'])
    num_clusters = list(df['cc_clusters'])
    mean_clusters = dic['cc_clusters']['mean'] 
    dist_threshold = 0.25*512*math.sqrt(2) # 25% of the diagonal of the image

    type_change = []
    '''
    If the largest component size is greater than 20% of the image size, then it is diffused
    If the number of clusters is greater than the mean number of clusters and the cluster distance is greater than the threshold, then it is diffused
    Else it is localized
    '''
    for i in tqdm(range(len(df)), total=len(df)):
        if(largest_comp_size[i]/(int(512*512)) > 0.2):
            type_change.append('diffused')
        elif(num_clusters[i] > mean_clusters) and (cluster_dist[i] > dist_threshold):
     
            type_change.append('diffused')
        else:
            type_change.append('localized')
    df['localization'] = type_change
    return df 

def sem_mag(df, col, dic):
    '''
    This function takes in the dataframe and the dictionary containing the values for the columns
    and returns the dataframe with the semantically meaningful column added
    Input:  df: dataframe
            dic: dictionary containing the values for the columns across 3 bins
    Output: df: dataframe with the semantically meaningful column added, small, medium or large
    '''
    small_max = dic[col]['small_threshold']
    large_max = dic[col]['large_threshold']
    col_name = col + '_category'
    data = df[col]
    method = df['method']
    type_change = []
    for i in tqdm(range(len(df)), total=len(df)):
        if(method[i] == 'p2p') and (col == 'ratio'):
            type_change.append('NA')
        else:
            if data[i] < small_max:
                type_change.append('small')
            elif data[i] <large_max:
                type_change.append('medium')
            else:
                type_change.append('large')
    df[col_name] = type_change

    return df 

def sem_mag_5(df, col, dic):
    '''
    This function takes in the dataframe and the dictionary containing the values for the columns
    and returns the dataframe with the semantically meaningful column added
    Input:  df: dataframe
            dic: dictionary containing the values for the columns across 5 bins
    Output: df: dataframe with the semantically meaningful column added, small, medium or large
    '''
    bin1 = dic[col]['bin1']
    bin2 = dic[col]['bin2']
    bin3= dic[col]['bin3']
    bin4 = dic[col]['bin4']
    col_name = col
    data = df[col]
    type_change = []
    for i in tqdm(range(len(df)), total=len(df)):
        if data[i] < bin1:
            type_change.append('1')
        elif data[i] <bin2:
            type_change.append('2')
        elif data[i] <bin3:
            type_change.append('3')
        elif data[i] <bin4:
            type_change.append('4')
        else:
            type_change.append('5')
    df[col_name] = type_change

    return df 
def get_value(obj):
    if isinstance(obj, list):
        return obj[0] 
    elif isinstance(obj, int):
        return obj

def scene_cols(df, df1):
    '''
    This function takes in the dataframe and the dictionary containing the values for the columns
    and returns the dataframe with the scene diversity and scene complexity columns added
    Input:  df: dataframe
            df1: dataframe containing the scene diversity and scene complexity values
    Output: df: dataframe with the scene diversity and scene complexity columns added
    '''
    img_id_df = list(df['image_id'])
    dataset = list(df['dataset'])
    img_id_df1 = list(df1['image_id'])
    scene_diversity,scene_complexity,scene_diversity_bins,scene_complexity_bins = [], [], [], []
    for i, img in enumerate(tqdm(img_id_df, total = len(img_id_df))):
        if(dataset[i] == 'CelebAHQ'):
            img = 'CelebA_' + str('{:05d}'.format(int(img)))
        try:
            idx_ = img_id_df1.index(img)
            idx = get_value(idx_)
            scene_diversity.append(df1['scene_diversity'][idx])
            scene_complexity.append(df1['scene_complexity'][idx])
            scene_diversity_bins.append(df1['scene_diversity_bins'][idx])
            scene_complexity_bins.append(df1['scene_complexity_bins'][idx])
        except:
            scene_diversity.append('NA')
            scene_complexity.append('NA')
            scene_diversity_bins.append('NA')
            scene_complexity_bins.append('NA')

    df['scene_diversity'] = scene_diversity
    df['scene_complexity'] = scene_complexity
    df['scene_diversity_category'] = scene_diversity_bins
    df['scene_complexity_category'] = scene_complexity_bins

    return df

parser = argparse.ArgumentParser(description='Create final bins')
parser.add_argument('--root_csv', type=str, required=True, help='path to the main csv file inpainting or prompt-based-editing or real images')
parser.add_argument('--save_csv', type=str, required=True, help='path to save the csv file')
parser.add_argument('--root_json', type=str, required=True, help='path to the json file containing the bin values')
parser.add_argument('--scene_csv', type=str, default=None, help='path to csv file containing the scene diversity and scene complexity values')

args = parser.parse_args()


df = pd.read_csv(args.root_csv)
if(args.scene_csv is not None):
    df_scene = pd.read_csv(args.scene_csv)
    df = scene_cols(df, df_scene)
    df.to_csv(args.save_csv, index=False)
    exit()
columns = ['post_edit_ratio', 'dreamsim','lpips_score','sen_sim','mse','ssim', 'area_ratio']
file_path = args.root_json

with open(file_path, 'r') as file:
    data = json.load(file)
for i in columns:
    df = sem_mag_5(df, i, data)

df = localized_diffused(df, data)


# df_merged = scene_cols(df, df1)
df = df.drop(columns=df.columns[[0]], axis=1)
df = df.drop(columns=['largest_component_size', 'cc_clusters', 'cluster_dist'])
df.to_csv(args.save_csv, index=False)


        
