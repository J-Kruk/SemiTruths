
import argparse
from magnitude_change import *
import pdb
from tqdm import tqdm
import os
import pdb 
parser = argparse.ArgumentParser(description="Color Scheme for Skyscapes and Uavid")
parser.add_argument('--root_csv', type=str, default='/srv/hoffman-lab/flash9/apal72/half-truths/Semi-Truths/metadata/edited/inpainting_2.csv',help='main csv') 
parser.add_argument('--root_dir', type=str, default='/srv/hoffman-lab/flash9/apal72/half-truths/Semi-Truths/inpainting',help='root directory to the images') 
parser.add_argument('--real_imgs', type=str, default='/srv/hoffman-lab/share4/apal72/half-truths/data/Half_Truths_Dataset/images/', help='path to images')
# parser.add_argument('--root_csv', type=str, default='/srv/hoffman-lab/share4/apal72/half-truths/data/Half_Truths_Dataset/generated_images/prompt-pompt/HF-set/metadata/edited/prompt-based-editing.csv',help='main csv') 
# parser.add_argument('--root_dir', type=str, default='/srv/hoffman-lab/share4/apal72/half-truths/data/Half_Truths_Dataset/generated_images/prompt-pompt/prompt-based-editing/prompt-based-editing',help='root directory to the images')      
parser.add_argument('--model', type=str, default='inpainting',help='model name') 
parser.add_argument('--save_dir', type=str, default=None) 


args = parser.parse_args() 
part = args.root_csv.split('.')[0].split('_')[-1]
print('PART--------', part)     

datasets = []
models = []
data = pd.read_csv(args.root_csv)

name = os.path.splitext(args.root_csv.split('/')[-1])[0] + '_mag-metrics_2.csv'
columns = ['post_edit_ratio', 'ssim', 'mse', 'lpips_score','dreamsim', 'sen_sim', 'largest_component_size', 'cc_clusters', 'cluster_dist']
real_paths = list(args.real_imgs+ data['dataset'].astype(str) +'/' + data['img_id'].astype(str))
if not(os.path.exists(os.path.join(args.root_dir,f'mag_csvs_{part}'))):
    os.makedirs(os.path.join(args.root_dir,f'mag_csvs_{part}'))
if(args.model == 'inpainting'):
    fake_paths = list(args.root_dir +'/'+ data['dataset'].astype(str) + '/' + data['diffusion_model'].astype(str) +'/'+data['perturbed_img_id'].astype(str)+ '_'+ data['dataset'].astype(str)+'_'+data['diffusion_model'].astype(str)+'.png' )
else:
    fake_paths = list(args.root_dir +'/'+ data['language_model'].astype(str)+'/'+ data['dataset'].astype(str) + '/' + data['diffusion_model'].astype(str) +'/'+data['perturbed_img_id'].astype(str)+ '_'+ data['dataset'].astype(str)+'_'+data['diffusion_model'].astype(str)+'.png' )

og_label = data['original_caption']
perturbed_label = data['perturbed_caption']
change_calc = Change(save_dir=args.save_dir)

metrics = []
for i in tqdm(range(len(real_paths)),total = len(real_paths)):
    if(os.path.exists(real_paths[i] + '.jpg')):
        real_paths[i] = real_paths[i] + '.jpg'
    else:
        real_paths[i] = real_paths[i] + '.png'
    result = change_calc.calc_metrics(real_paths[i], fake_paths[i],og_label[i],perturbed_label[i])
    metrics.append(result)
    if((i+1)%2000 == 0):

        na_len = len(data) - len(metrics) 
        metrics_rest = [['NA']*len(columns)]*na_len
        metrics_all = metrics + metrics_rest
        metrics_data = pd.DataFrame(metrics_all, columns=columns)
        final_data = pd.concat([data, metrics_data], axis=1)
        final_data.to_csv(os.path.join(args.root_dir,f'mag_csvs_{part}',name), index=False)


columns = ['post_edit_ratio', 'ssim', 'mse', 'lpips_score','dreamsim', 'sen_sim', 'largest_component_size', 'cc_cluters', 'cluster_dist']
metrics_data = pd.DataFrame(metrics, columns=columns)

final_data = pd.concat([data, metrics_data], axis=1)

name = os.path.splitext(args.root_csv.split('/')[-1])[0] + '_mag-metrics_final.csv'
if not(os.path.exists(os.path.join(args.root_dir,f'mag_csvs_{part}'))):
    os.makedirs(os.path.join(args.root_dir,f'mag_csvs_{part}'))
final_data.to_csv(os.path.join(args.root_dir,f'mag_csvs_{part}',name), index=False)





