import pandas as pd
import json
import math
from collections import Counter
import os
import sys 
import pdb
from tqdm import tqdm 
def localized_diffused(df, dic):
    
    largest_comp_size = list(df['largest_component_size'])
    cluster_dist = list(df['cluster_dist'])
    num_clusters = list(df['cc_clusters'])
    mean_clusters = dic['cc_clusters']['mean']
    dist_threshold = 0.25*512*math.sqrt(2)

    type_change = []
    for i in tqdm(range(len(df)), total=len(df)):
        if(largest_comp_size[i]/(int(512*512)) > 0.2):
            type_change.append('diffused')
        elif(num_clusters[i] > mean_clusters) and (cluster_dist[i] > dist_threshold):
     
            type_change.append('diffused')
        else:
            # pdb.set_trace()
            type_change.append('localized')
    df['localization'] = type_change
    return df 

def sem_mag(df, col, dic):
    # pdb.set_trace()
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
    # pdb.set_trace()
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

def majority_vote(row):
    # pdb.set_trace()
    counter = Counter(row)
    most_common = counter.most_common(1)
    return most_common[0][0]

def conditional_majority_vote(row):

    if row['method'] == 'inpainting':
        columns_to_consider = ['ratio_category','ratio_rgb_category','ssim_rgb_category']
        return majority_vote(row[columns_to_consider])
    else:
        columns_to_consider = ['ratio_rgb_category','ssim_rgb_category']
        return majority_vote(row[columns_to_consider])

def scene_cols(df, df1):
    # pdb.set_trace()
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



df = pd.read_csv('/srv/hoffman-lab/flash9/apal72/half-truths/Semi-Truths/metadata/edited/bins/inpainting.csv')
# df1 = pd.read_csv('/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/metadata_flat_scene_diversity_complexity.csv')
# columns_to_consider_sem = ['dreamsim_category','lpips_score_category','sen_sim_category']
columns = ['post_edit_ratio', 'dreamsim','lpips_score','sen_sim','mse','ssim', 'area_ratio']
file_path = '/srv/hoffman-lab/flash9/apal72/half-truths/Semi-Truths/inpainting/mag_vals_5.json'

with open(file_path, 'r') as file:
    data = json.load(file)
for i in columns:
    df = sem_mag_5(df, i, data)

df = localized_diffused(df, data)

# df['sem_mag_category'] = df[columns_to_consider_sem].apply(majority_vote, axis=1)
# df['size_mag_category'] = df.apply(conditional_majority_vote, axis=1)


# df_merged = scene_cols(df, df1)
pdb.set_trace()
df = df.drop(columns=df.columns[[0]], axis=1)
df = df.drop(columns=['largest_component_size', 'cc_clusters', 'cluster_dist'])
path = '/srv/hoffman-lab/flash9/apal72/half-truths/Semi-Truths/metadata/edited/bins/inpainting.csv'
df.to_csv(path)


        