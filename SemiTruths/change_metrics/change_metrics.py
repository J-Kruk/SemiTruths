
import argparse
from magnitude_change import *
import pdb
from tqdm import tqdm
import os
import pdb 

'''
This script calculates the raw values of the change metrics for all the perturbed images in the dataset. The change metrics include the following:
1. Post Edit Ratio
2. SSIM
3. MSE
4. LPIPS Score
5. Dreamsim
6. Sen_sim
7. Largest Component Size
8. Connected Components
9. Cluster Distance
Example Usage: python change_metrics.py --root_csv /path/to/main/csv --root_dir /path/to/perturbed/images --real_imgs /path/to/real/images --model inpainting --save_dir /path/to/save/directory
'''

parser = argparse.ArgumentParser(description="Color Scheme for Skyscapes and Uavid")
parser.add_argument('--root_csv', type=str, required=True,help='path to main csv containing all the metadata  details for all the perturbed images') 
parser.add_argument('--root_dir', type=str, required=True,help='root directory to the perturbed images') 
parser.add_argument('--real_imgs', type=str, required=True, help='root directory to real images')
parser.add_argument('--model', type=str, default='inpainting',help='perturbation type') 
parser.add_argument('--save_dir', type=str, default=None, help='path where change metrics csvs should be saved') 


args = parser.parse_args() 
part = args.root_csv.split('.')[0].split('_')[-1] #Name of the csv file
datasets = [] #List of all the datasets
models = [] #List of all the models
data = pd.read_csv(args.root_csv) #Reading the csv file

name = os.path.splitext(args.root_csv.split('/')[-1])[0] + '_mag-metrics.csv' #Name of the root csv file
columns = ['post_edit_ratio', 'ssim', 'mse', 'lpips_score','dreamsim', 'sen_sim', 'largest_component_size', 'cc_clusters', 'cluster_dist'] #Columns for the change metrics csv file
real_paths = list(args.real_imgs+ data['dataset'].astype(str) +'/' + data['img_id'].astype(str)) #List of all the real images


if not(os.path.exists(os.path.join(args.save_dir,f'mag_csvs_{part}'))): #Creating a directory to save the intermediate change metrics csv files
    os.makedirs(os.path.join(args.save_dir,f'mag_csvs_{part}'))

# Reading the list of fake images
if(args.model == 'inpainting'):
    fake_paths = list(args.root_dir +'/'+ data['dataset'].astype(str) + '/' + data['diffusion_model'].astype(str) +'/'+data['perturbed_img_id'].astype(str)+ '_'+ data['dataset'].astype(str)+'_'+data['diffusion_model'].astype(str)+'.png' )
else:
    fake_paths = list(args.root_dir +'/'+ data['language_model'].astype(str)+'/'+ data['dataset'].astype(str) + '/' + data['diffusion_model'].astype(str) +'/'+data['perturbed_img_id'].astype(str)+ '_'+ data['dataset'].astype(str)+'_'+data['diffusion_model'].astype(str)+'.png' )

og_label = data['original_caption'] #Original caption of the image
perturbed_label = data['perturbed_caption'] #Perturbed caption of the image
change_calc = Change(save_dir=args.save_dir) #Initializing the change metrics class

metrics = [] #List to store the change metrics
'''
Calculating the change metrics for all the images
'''
for i in tqdm(range(len(real_paths)),total = len(real_paths)):
    if(os.path.exists(real_paths[i] + '.jpg')):
        real_paths[i] = real_paths[i] + '.jpg'
    else:
        real_paths[i] = real_paths[i] + '.png'
    result = change_calc.calc_metrics(real_paths[i], fake_paths[i],og_label[i],perturbed_label[i]) #Calculating the change metrics
    metrics.append(result)

    '''
    Saving the change metrics in the csv file after every 2000 images
    '''
    if((i+1)%2000 == 0):

        na_len = len(data) - len(metrics) #Calculating the number of rows which are not yet calculated
        metrics_rest = [['NA']*len(columns)]*na_len #Filling the rest of the rows with NA
        metrics_all = metrics + metrics_rest #Concatenating the calculated and rest of the rows
        metrics_data = pd.DataFrame(metrics_all, columns=columns) #Creating a dataframe of the change metrics
        final_data = pd.concat([data, metrics_data], axis=1) #Concatenating the original data and the change metrics
        final_data.to_csv(os.path.join(args.save_dir,f'mag_csvs_{part}',name), index=False) #Saving the change metrics csv file


'''
Saving the final change metrics csv file
'''

metrics_data = pd.DataFrame(metrics, columns=columns) 
final_data = pd.concat([data, metrics_data], axis=1)
name = os.path.splitext(args.root_csv.split('/')[-1])[0] + '_mag-metrics_final.csv'
if not(os.path.exists(os.path.join(args.save_dir,f'mag_csvs_{part}'))):
    os.makedirs(os.path.join(args.save_dir,f'mag_csvs_{part}'))
final_data.to_csv(os.path.join(args.save_dir,f'mag_csvs_{part}',name), index=False)





