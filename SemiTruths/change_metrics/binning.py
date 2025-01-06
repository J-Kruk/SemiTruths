import pandas as pd 
import glob 
import pdb
import json 
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import numpy as np
import pdb


parser = argparse.ArgumentParser(description="Creating bin thresholds")
parser.add_argument('--inpainting_csv', type=str, required=True,help='path to main csv containing all change metrics values for the inpainting perturbed images') 
parser.add_argument('--p2p_csv', type=str, required=True,help='path to main csv containing all change metrics values for the prompt-based-editing perturbed images') 
parser.add_argument('--save_dir', type=str, required=True,help='path to save the json file') 
args = parser.parse_args()

def convert_to_native_types(data):
    '''
    Convert numpy types to native python types
    '''
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(element) for element in data]
    else:
        return data


def find_diff(df, col1, col2):
    '''
    Find the maximum and minimum difference between two columns
    '''
    difference= (df[col1]- df[col2]).tolist()
    absolute_difference = [abs(num) for num in difference]
    max_val = max(absolute_difference)
    min_val = min(absolute_difference)
    return max_val, min_val

def find_corr(df,col1,col2):
    '''
    Find the correlation between two columns
    '''
    return df[col1].corr(df[col2])

def create_bins(df, col, file_path):
    '''
    Create bins for the columns
    If the column is a metric, then the bins are created based on the 25th and 75th percentile
    If the column is a count, then the bins are created based on the mean
    Inputs: df: dataframe
            col: column name
            file_path: path to save the json file
    Outputs: max_val: maximum value of the column
    '''

    max_val = df[col].max()
    min_val = df[col].min()
    if(col == 'cc_clusters') or (col =='cluster_dist') or (col == 'largest_component_size'):
        mean_val = bin_values(df[col], col)

        return  max_val, min_val, mean_val
    else:
        small_max, large_max = bin_values(df[col], col)
        return max_val, min_val, small_max, large_max

def create_bins_5(df, col, file_path):
   '''
    Create bins for the columns
    If the column is a metric, then the bins are created based on the 20th, 40th, 60th and 80th percentile
    If the column is a count, then the bins are created based on the mean
    Inputs: df: dataframe
            col: column name
            file_path: path to save the json file
    Outputs: max_val: maximum value of the column
    '''

    max_val = df[col].max()
    min_val = df[col].min()
    if(col == 'cc_clusters') or (col =='cluster_dist') or (col == 'largest_component_size'):
        mean_val = bin_values_5(df[col], col)
        return  max_val, min_val, mean_val
    else:
        bin_1, bin2, bin3, bin4 = bin_values_5(df[col], col)
        return max_val, min_val, bin_1, bin2, bin3, bin4

def bin_values(values, col):
    '''
    Define the bin boundaries
    Inputs: values: list of values
            col: column name
    Outputs: small_max: 25th percentile
             large_max: 75th percentile
    '''
    small_max = np.percentile(np.array(values), 25)
    large_max = np.percentile(np.array(values), 75)
    if(col == 'cc_clusters') or (col =='cluster_dist') or (col == 'largest_component_size'):
        return np.mean(np.array(df[col]))
    else:
        return small_max, large_max


def bin_values_5(values, col):
    '''
    Define the bin boundaries
    Inputs: values: list of values
            col: column name
    Outputs: bin1: 20th percentile
             bin2: 40th percentile
             bin3: 60th percentile
             bin4: 80th percentile
    '''

    bin1 = np.percentile(np.array(values), 20)
    bin2 = np.percentile(np.array(values), 40)
    bin3 = np.percentile(np.array(values), 60)
    bin4 = np.percentile(np.array(values), 80)
    if(col == 'cc_clusters') or (col =='cluster_dist') or (col == 'largest_component_size'):
        return np.mean(np.array(df[col]))
    else:
        return bin1, bin2, bin3, bin4

def diff_sem(df,dic, file_path):
    '''
    Create 3 bins for all the columns 
    Inputs: df: dataframe
            dic: dictionary to store the bin values
            file_path: path to save the json file
    Outputs: dic: dictionary with bin values
    '''
   
    dic = {}
    
    print('DREAMSIM')
    dic['dreamsim'] = {}
    dic['dreamsim']['max'], dic['dreamsim']['min'], dic['dreamsim']['small_threshold'], dic['dreamsim']['large_threshold'] = create_bins(df,'dreamsim',file_path)
    
    print('LPIPS')
    dic['lpips_score'] = {}
    dic['lpips_score']['max'], dic['lpips_score']['min'], dic['lpips_score']['small_threshold'], dic['lpips_score']['large_threshold'] = create_bins(df,'lpips_score',file_path)
    
    print('SEN_SIM')
    dic['sen_sim'] = {}
    dic['sen_sim']['max'], dic['sen_sim']['min'], dic['sen_sim']['small_threshold'], dic['sen_sim']['large_threshold'] = create_bins(df,'sen_sim',file_path)

    
    print('MSE')
    dic['mse'] = {}
    dic['mse']['max'], dic['mse']['min'],dic['mse']['small_threshold'], dic['mse']['large_threshold'] = create_bins(df,'mse',file_path)

    
    print('SSIM')
    dic['ssim'] = {}
    dic['ssim']['max'], dic['ssim']['min'], dic['ssim']['small_threshold'], dic['ssim']['large_threshold'] = create_bins(df,'ssim',file_path)

    
    print('RATIO')
    dic['post_edit_ratio'] = {}
    dic['post_edit_ratio']['max'], dic['post_edit_ratio']['min'], dic['post_edit_ratio']['small_threshold'], dic['post_edit_ratio']['large_threshold'] = create_bins(df,'post_edit_ratio',file_path)


    print('CC')
    dic['largest_component_size'] = {}
    dic['largest_component_size']['max'], dic['largest_component_size']['min'], dic['largest_component_size']['mean'] = create_bins(df,'largest_component_size',file_path)

    dic['cc_clusters'] = {}
    dic['cc_clusters']['max'], dic['cc_clusters']['min'], dic['cc_clusters']['mean'] = create_bins(df,'cc_clusters',file_path)

    dic['cluster_dist'] = {}
    dic['cluster_dist']['max'], dic['cluster_dist']['min'],dic['cluster_dist']['mean'] = create_bins(df,'cluster_dist',file_path)

    return dic


def diff_sem_5(df, df_2, dic, file_path):

    '''
    Create 5 bins for all the columns
    Inputs: df: combined dataframe
            df_2: inpainting dataframe 
            dic: dictionary to store the bin values
            file_path: path to save the json file
    Outputs: dic: dictionary with bin values
    '''
    dic = {}
    

    dic['dreamsim'] = {}
    dic['dreamsim']['max'], dic['dreamsim']['min'], dic['dreamsim']['bin1'], dic['dreamsim']['bin2'], dic['dreamsim']['bin3'], dic['dreamsim']['bin4'] = create_bins_5(df,'dreamsim',file_path)
    
    print('LPIPS')
    dic['lpips_score'] = {}
    dic['lpips_score']['max'], dic['lpips_score']['min'], dic['lpips_score']['bin1'], dic['lpips_score']['bin2'],  dic['lpips_score']['bin3'],  dic['lpips_score']['bin4'] = create_bins_5(df,'lpips_score',file_path)
    
    print('SEN_SIM')
    dic['sen_sim'] = {}
    dic['sen_sim']['max'], dic['sen_sim']['min'], dic['sen_sim']['bin1'], dic['sen_sim']['bin2'], dic['sen_sim']['bin3'], dic['sen_sim']['bin4'] = create_bins_5(df,'sen_sim',file_path)
    
    
    print('MSE')
    dic['mse'] = {}
    dic['mse']['max'], dic['mse']['min'],dic['mse']['bin1'], dic['mse']['bin2'], dic['mse']['bin3'], dic['mse']['bin4'] = create_bins_5(df,'mse',file_path)
    
    print('SSIM')
    dic['ssim'] = {}
    dic['ssim']['max'], dic['ssim']['min'], dic['ssim']['bin1'], dic['ssim']['bin2'], dic['ssim']['bin3'], dic['ssim']['bin4'] = create_bins_5(df,'ssim',file_path)

    
    print('RATIO')
    dic['post_edit_ratio'] = {}
    dic['post_edit_ratio']['max'], dic['post_edit_ratio']['min'], dic['post_edit_ratio']['bin1'], dic['post_edit_ratio']['bin2'], dic['post_edit_ratio']['bin3'], dic['post_edit_ratio']['bin4'] = create_bins_5(df,'post_edit_ratio',file_path)
    
    
    print('CC')
    dic['largest_component_size'] = {}
    dic['largest_component_size']['max'], dic['largest_component_size']['min'], dic['largest_component_size']['mean'] = create_bins_5(df,'largest_component_size',file_path)

    dic['cc_clusters'] = {}
    dic['cc_clusters']['max'], dic['cc_clusters']['min'], dic['cc_clusters']['mean'] = create_bins_5(df,'cc_clusters',file_path)

    dic['cluster_dist'] = {}
    dic['cluster_dist']['max'], dic['cluster_dist']['min'],dic['cluster_dist']['mean'] = create_bins_5(df,'cluster_dist',file_path)

    dic['area_ratio'] = {}
    dic['area_ratio']['max'], dic['area_ratio']['min'], dic['area_ratio']['bin1'], dic['area_ratio']['bin2'], dic['area_ratio']['bin3'], dic['area_ratio']['bin4'] = create_bins_5(df_2,'area_ratio',file_path)

    return dic

def print_vals(combined_df):

    max_dream_lpips,min_dream_lpips = find_diff(combined_df,'dreamsim','lpips_score')
    cor_dream_lpips = find_corr(combined_df,'dreamsim','lpips_score')

    max_sen_clip, min_sen_clip = find_diff(combined_df,'sen_sim','clip_sim')
    cor_sen_clip = find_corr(combined_df,'sen_sim','clip_sim')

    max_dif_rgb_gray, min_dif_rgb_gray = find_diff(combined_df,'mse','mse_gray')
    cor_rgb_gray = find_corr(combined_df,'mse','mse_gray')

    max_dif_post_edit_ratio_gray, min_dif_post_edit_ratio_gray = find_diff(combined_df,'post_edit_ratio','ratio_gray')
    cor_post_edit_ratio_gray = find_corr(combined_df,'post_edit_ratio','ratio_gray')

    max_dif_ratio_mse, min_dif_ratio_mse = find_diff(combined_df,'mse','post_edit_ratio')
    cor_ratio_mse = find_corr(combined_df,'post_edit_ratio','mse')

    print('DREAM_LPIPS MAX: ',max_dream_lpips, 'MIN : ', min_dream_lpips, 'COR: ', cor_dream_lpips)
    print('SEN_CLIP MAX: ',max_sen_clip, 'MIN : ', min_sen_clip, 'COR: ', cor_sen_clip)
    print('mse_GRAY MAX: ',max_dif_rgb_gray, 'MIN : ', min_dif_rgb_gray, 'COR: ', cor_rgb_gray)
    print('post_edit_ratio_GRAY MAX: ',max_dif_post_edit_ratio_gray, 'MIN : ', min_dif_post_edit_ratio_gray, 'COR: ', cor_post_edit_ratio_gray)
    print('RATIO_mse MAX: ',max_dif_ratio_mse, 'MIN : ', min_dif_ratio_mse, 'COR: ', cor_ratio_mse)

file_path=args.save_dir
combined_data = []
columns_dtype = {
    'img_id': str,
    'perturbed_img_id': str,
    'original_caption': str,
    'perturbed_caption': str,
    'dataset': str,
    'diffusion_model': str,
    'sem_magnitude': str,
    'post_edit_ratio': float,
    'ssim': float,
    'mse': float,
    'lpips_score': float,
    'dreamsim': float,
    'sen_sim': float,
    'largest_component_size': float,
    'cc_clusters': float,
    'cluster_dist': float
}

columns_dtype_inpainting = {
    'img_id': str,
    'perturbed_img_id': str,
    'original_caption': str,
    'perturbed_caption': str,
    'dataset': str,
    'diffusion_model': str,
    'sem_magnitude': str,
    'post_edit_ratio': float,
    'ssim': float,
    'mse': float,
    'lpips_score': float,
    'dreamsim': float,
    'sen_sim': float,
    'largest_component_size': float,
    'cc_clusters': float,
    'cluster_dist': float,
    'area_ratio': float
}

'''

Read the csvs and clean the data
'''
df_p2p = pd.read_csv(args.p2p_csv, dtype=columns_dtype)
df_inpainting = pd.read_csv(args.inpainting_csv, dtype=columns_dtype)
df_inpainting = df_inpainting.drop(columns=['mask_id', 'mask_name', 'area_ratio'])
df = pd.concat([df_p2p, df_inpainting], axis=0, ignore_index=True)
df_inpainting_new = pd.read_csv(args.inpainting_csv, dtype=columns_dtype_inpainting)

'''
Filter the data
'''
df = df[df['pass_qc'] == 1]
df_inpainting_new = df_inpainting_new[df_inpainting_new['pass_qc'] == 1]
df_cleaned = df.replace('NA', pd.NA).dropna()
df_inpainting_cleaned = df_inpainting_new.replace('NA', pd.NA).dropna()

'''
Create bins for the columns
'''
dic_5 = {}
dic_5 = diff_sem_5(df_cleaned, df_inpainting_cleaned, dic_5,file_path)
native_data = convert_to_native_types(dic_5)

'''
Save the json file
'''
with open(file_path +'/mag_vals_5.json', "w") as json_file:
    json.dump(native_data, json_file)


