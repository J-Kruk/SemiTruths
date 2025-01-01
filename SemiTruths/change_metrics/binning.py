import pandas as pd 
import glob 
import pdb
import json 
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import pdb
root_path = '/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/mag_csvs'
all_csvs = glob.glob(root_path+'/*_2.csv')

def convert_to_native_types(data):
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
    # pdb.set_trace()
    difference= (df[col1]- df[col2]).tolist()
    absolute_difference = [abs(num) for num in difference]
    max_val = max(absolute_difference)
    min_val = min(absolute_difference)
    return max_val, min_val

def find_corr(df,col1,col2):
    return df[col1].corr(df[col2])

def create_bins(df, col, file_path,var='whole'):
    # pdb.set_trace()

    max_val = df[col].max()
    min_val = df[col].min()
    # plt.hist(df[col].tolist(), bins=10000, edgecolor='moccasin')
    # # Add labels and title
    # plt.xlabel('Magnitude of Change')
    # plt.ylabel('Frequency')

    # plt.title(col+' distribution')

    # # Show the plot
    # if(not(os.path.exists(os.path.join(file_path, var)))):
    #     os.makedirs(os.path.join(file_path, var))
    if(col == 'cc_clusters') or (col =='cluster_dist') or (col == 'largest_component_size'):
        mean_val = bin_values(df[col], col)

        return  max_val, min_val, mean_val
    else:
        small_max, large_max = bin_values(df[col], col)
        return max_val, min_val, small_max, large_max

def create_bins_5(df, col, file_path,var='whole'):
   

    max_val = df[col].max()
    min_val = df[col].min()
    if(col == 'cc_clusters') or (col =='cluster_dist') or (col == 'largest_component_size'):
        mean_val = bin_values_5(df[col], col)
        return  max_val, min_val, mean_val
    else:
        bin_1, bin2, bin3, bin4 = bin_values_5(df[col], col)
        return max_val, min_val, bin_1, bin2, bin3, bin4

def bin_values(values, col):
    # Define the bin boundaries
    # pdb.set_trace()
    small_max = np.percentile(np.array(values), 25)
    large_max = np.percentile(np.array(values), 75)
    if(col == 'cc_clusters') or (col =='cluster_dist') or (col == 'largest_component_size'):
        return np.mean(np.array(df[col]))
    else:
        return small_max, large_max


def bin_values_5(values, col):
    # Define the bin boundaries

    bin1 = np.percentile(np.array(values), 20)
    bin2 = np.percentile(np.array(values), 40)
    bin3 = np.percentile(np.array(values), 60)
    bin4 = np.percentile(np.array(values), 80)
    if(col == 'cc_clusters') or (col =='cluster_dist') or (col == 'largest_component_size'):
        return np.mean(np.array(df[col]))
    else:
        return bin1, bin2, bin3, bin4
    # for i in values:
    #     if(i < small_max):
    #         change.append('low')
    #     elif(small_max <= i < medium_max):
    #         change.append('medium')
    #     else:
    #         change.append('high')
    
    # return scene, small_max, medium_max

def diff_sem(df,dic, file_path, var='whole'):
   
    dic = {}
    
    print('DREAMSIM')
    dic['dreamsim'] = {}
    dic['dreamsim']['max'], dic['dreamsim']['min'], dic['dreamsim']['small_threshold'], dic['dreamsim']['large_threshold'] = create_bins(df,'dreamsim',file_path, var)
    
    print('LPIPS')
    dic['lpips_score'] = {}
    dic['lpips_score']['max'], dic['lpips_score']['min'], dic['lpips_score']['small_threshold'], dic['lpips_score']['large_threshold'] = create_bins(df,'lpips_score',file_path, var)
    
    print('SEN_SIM')
    dic['sen_sim'] = {}
    dic['sen_sim']['max'], dic['sen_sim']['min'], dic['sen_sim']['small_threshold'], dic['sen_sim']['large_threshold'] = create_bins(df,'sen_sim',file_path, var)

    
    print('MSE')
    dic['mse'] = {}
    dic['mse']['max'], dic['mse']['min'],dic['mse']['small_threshold'], dic['mse']['large_threshold'] = create_bins(df,'mse',file_path, var)

    
    print('SSIM')
    dic['ssim'] = {}
    dic['ssim']['max'], dic['ssim']['min'], dic['ssim']['small_threshold'], dic['ssim']['large_threshold'] = create_bins(df,'ssim',file_path,var)

    
    print('RATIO')
    dic['post_edit_ratio'] = {}
    dic['post_edit_ratio']['max'], dic['post_edit_ratio']['min'], dic['post_edit_ratio']['small_threshold'], dic['post_edit_ratio']['large_threshold'] = create_bins(df,'post_edit_ratio',file_path,var)


    print('CC')
    dic['largest_component_size'] = {}
    dic['largest_component_size']['max'], dic['largest_component_size']['min'], dic['largest_component_size']['mean'] = create_bins(df,'largest_component_size',file_path,var)

    dic['cc_clusters'] = {}
    dic['cc_clusters']['max'], dic['cc_clusters']['min'], dic['cc_clusters']['mean'] = create_bins(df,'cc_clusters',file_path,var)

    dic['cluster_dist'] = {}
    dic['cluster_dist']['max'], dic['cluster_dist']['min'],dic['cluster_dist']['mean'] = create_bins(df,'cluster_dist',file_path,var)

    return dic


def diff_sem_5(df, df_2, dic, file_path, var='whole'):

    dic = {}
    # if(var == 'whole'):
    #     df = df_all
    # else:
    #     df = (df_all[df_all['sem_magnitude'] == var]).reset_index(drop=True)
    

    dic['dreamsim'] = {}
    dic['dreamsim']['max'], dic['dreamsim']['min'], dic['dreamsim']['bin1'], dic['dreamsim']['bin2'], dic['dreamsim']['bin3'], dic['dreamsim']['bin4'] = create_bins_5(df,'dreamsim',file_path, var)
    
    print('LPIPS')
    dic['lpips_score'] = {}
    dic['lpips_score']['max'], dic['lpips_score']['min'], dic['lpips_score']['bin1'], dic['lpips_score']['bin2'],  dic['lpips_score']['bin3'],  dic['lpips_score']['bin4'] = create_bins_5(df,'lpips_score',file_path, var)
    
    print('SEN_SIM')
    dic['sen_sim'] = {}
    dic['sen_sim']['max'], dic['sen_sim']['min'], dic['sen_sim']['bin1'], dic['sen_sim']['bin2'], dic['sen_sim']['bin3'], dic['sen_sim']['bin4'] = create_bins_5(df,'sen_sim',file_path, var)
    
    # print('CLIP_SIM')
    # dic['clip_sim'] = {}
    # dic['clip_sim']['max'], dic['clip_sim']['min'], dic['clip_sim']['bin1'], dic['clip_sim']['bin2'], dic['clip_sim']['bin3'], dic['clip_sim']['bin4']   = create_bins_5(df,'clip_sim',file_path, var)
    
    print('MSE')
    dic['mse'] = {}
    dic['mse']['max'], dic['mse']['min'],dic['mse']['bin1'], dic['mse']['bin2'], dic['mse']['bin3'], dic['mse']['bin4'] = create_bins_5(df,'mse',file_path, var)
    # dic['mse_gray'] = {}
    # dic['mse_gray']['max'], dic['mse_gray']['min'], dic['mse_gray']['bin1'], dic['mse_gray']['bin2'], dic['mse_gray']['bin3'], dic['mse_gray']['bin4'] = create_bins_5(df,'mse_gray',file_path,var)
    
    print('SSIM')
    dic['ssim'] = {}
    dic['ssim']['max'], dic['ssim']['min'], dic['ssim']['bin1'], dic['ssim']['bin2'], dic['ssim']['bin3'], dic['ssim']['bin4'] = create_bins_5(df,'ssim',file_path,var)
    # dic['ssim_gray'] = {}
    # dic['ssim_gray']['max'], dic['ssim_gray']['min'], dic['ssim_gray']['bin1'], dic['ssim_gray']['bin2'], dic['ssim_gray']['bin3'], dic['ssim_gray']['bin4']= create_bins_5(df,'ssim_gray',file_path,var)
    
    print('RATIO')
    dic['post_edit_ratio'] = {}
    dic['post_edit_ratio']['max'], dic['post_edit_ratio']['min'], dic['post_edit_ratio']['bin1'], dic['post_edit_ratio']['bin2'], dic['post_edit_ratio']['bin3'], dic['post_edit_ratio']['bin4'] = create_bins_5(df,'post_edit_ratio',file_path,var)
    
    # dic['ratio_gray'] = {}
    # dic['ratio_gray']['max'], dic['ratio_gray']['min'], dic['ratio_gray']['bin1'], dic['ratio_gray']['bin2'], dic['ratio_gray']['bin3'], dic['ratio_gray']['bin4']= create_bins_5(df,'ratio_gray',file_path,var)
    
    print('CC')
    dic['largest_component_size'] = {}
    dic['largest_component_size']['max'], dic['largest_component_size']['min'], dic['largest_component_size']['mean'] = create_bins_5(df,'largest_component_size',file_path,var)

    dic['cc_clusters'] = {}
    dic['cc_clusters']['max'], dic['cc_clusters']['min'], dic['cc_clusters']['mean'] = create_bins_5(df,'cc_clusters',file_path,var)

    dic['cluster_dist'] = {}
    dic['cluster_dist']['max'], dic['cluster_dist']['min'],dic['cluster_dist']['mean'] = create_bins_5(df,'cluster_dist',file_path,var)

    dic['area_ratio'] = {}
    dic['area_ratio']['max'], dic['area_ratio']['min'], dic['area_ratio']['bin1'], dic['area_ratio']['bin2'], dic['area_ratio']['bin3'], dic['area_ratio']['bin4'] = create_bins_5(df_2,'area_ratio',file_path,var)

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

file_path='/srv/hoffman-lab/flash9/apal72/half-truths/Semi-Truths/inpainting'
# filename = '/srv/hoffman-lab/flash9/apal72/half-truths/Semi-Truths/metadata/'
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

columns_dtype_2 = {
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

# for filename in all_csvs:
df1 = pd.read_csv('/srv/hoffman-lab/share4/apal72/half-truths/data/Half_Truths_Dataset/generated_images/prompt-pompt/prompt-based-editing/prompt-based-editing/metadata/edited/bins/prompt_based_editing.csv', dtype=columns_dtype)
df2 = pd.read_csv('/srv/hoffman-lab/flash9/apal72/half-truths/Semi-Truths/metadata/edited/bins/inpainting.csv', dtype=columns_dtype)
df2 = df2.drop(columns=['mask_id', 'mask_name', 'area_ratio'])
df = pd.concat([df1, df2], axis=0, ignore_index=True)
df_new = pd.read_csv('/srv/hoffman-lab/flash9/apal72/half-truths/Semi-Truths/metadata/edited/bins/inpainting.csv', dtype=columns_dtype_2)

# pass_qc = list(df_pass_qc['pass_qc'])
# img_id_main = list(df['perturbed_img_id'])
# img_id_perturbed = list(df_pass_qc['perturbed_img_id'])
# pass_qc_main = []
# for img_id in tqdm(img_id_main, total=len(img_id_main)):
#     try:
#         pass_qc_main.append(pass_qc[img_id_perturbed.index(img_id)])
#     except:
#         pass_qc_main.append('NA')

# df['pass_qc'] = pass_qc_main
# combined_data = []
# for i in range(1,5):
#     df_temp = pd.read_csv(f'{file_path}/mag_csvs_{i}/part_{i}_mag-metrics_final.csv', dtype=columns_dtype)
#     # df_temp = df_temp['img_id'].astype(str)
#     df = pd.concat([df_temp, df], ignore_index=True)

df = df[df['pass_qc'] == 1]
df_new = df_new[df_new['pass_qc'] == 1]
df_cleaned = df.replace('NA', pd.NA).dropna()
df_new_cleaned = df_new.replace('NA', pd.NA).dropna()
# combined_df = pd.DataFrame(combined_data, columns=combined_columns.columns)
# combined_inpainting = []

# for filename in all_csvs:
#     df = pd.read_csv(filename)
#     if not(filename.split('/')[-1] == 'final_filtered_sample_data_mag-metrics_2.csv'):
#         # df = df['ratio']
#         combined_inpainting.extend(df.values.tolist())
# combined_df = pd.DataFrame(combined_inpainting, columns=df.columns)

# dic = {}
# dic = diff_sem(df_cleaned,dic,file_path,'whole')

dic_5 = {}
dic_5 = diff_sem_5(df_cleaned, df_new_cleaned, dic_5,file_path,'whole')
native_data = convert_to_native_types(dic_5)
# with open(file_path +'/mag_vals.json', "w") as json_file:
#     json.dump(native_data, json_file)

with open(file_path +'/mag_vals_5.json', "w") as json_file:
    json.dump(native_data, json_file)



