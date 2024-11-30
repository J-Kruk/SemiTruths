import torch
from PIL import Image
from utils import *
from tqdm import tqdm
import pdb
from tqdm import tqdm
import csv
import os

torch.set_default_tensor_type('torch.cuda.FloatTensor') 

DS = 'SUN_RGBD'
MODEL = "StableDiffusion_v5"

CSV_READ_FILE = "" # metadata for generated images

column_map ={
    'image':0,
    'mask':1,
    'img_id':2,
    'mask_id':3,
    'mask_name':4,
    'ratio':5,
    'entities':6,
    'classes':7,
    'perturbed_label':8,
    'sem_magnitude':9,
    'quality_flag':10,
    'dataset':11,
    'method':12,
    'model':13,
    'perturbed_path':14
}

DS_extemsion_dict ={
        "ADE20K": ".jpg",
        "CelebAHQ": ".jpg",
        "CityScapes": ".png",
        "HumanParsing": ".png",
        "OpenImages": ".jpg",
        "SUN_RGBD": ".jpg",
    }

PATH_TO_DATA_PARENT = "" #path to original images parent directory
CSV_POSTGEN_QC = "" #path to save postgen quality check results
PATH_TO_PERTURB_DATA_PARENT = "" #path to perturbed images parent directory
EDIT_EXTENSION = ".png"

def post_qual_check_row(row, 
                           writer, 
                           DS, 
                           column_map, 
                           PATH_TO_DATA_PARENT, 
                           PATH_TO_PERTURB_DATA_PARENT, 
                           DS_EXTENSION, 
                           EDIT_EXTENSION
                           ):

    if len(row[column_map['image']]):
            method = row[column_map['method']]
            model = row[column_map['model']]
            
            # remove model from orig path when just running independently
            # orig_img_path = os.path.join(PATH_TO_DATA_PARENT, DS,model, row[column_map['img_id']]+DS_EXTENSION)
            try:
                orig_img_path = os.path.join(PATH_TO_DATA_PARENT, DS, row[column_map['img_id']]+DS_EXTENSION)
                orig_img = Image.open(orig_img_path)
                orig_caption = row[column_map['mask_name']]
                if len(orig_img.getbands())!=3:
                    return
                brisque_score_orig = brisque_Score(orig_img)
                #if perturbed image path does not exist, return:
                if not os.path.exists(os.path.join(PATH_TO_PERTURB_DATA_PARENT , row[column_map['dataset']], model ,row[column_map['mask_id']]+"_"+DS +"_"+model + EDIT_EXTENSION)):
                    print("perturbed image does not exist")
                    print(os.path.join(PATH_TO_PERTURB_DATA_PARENT , row[column_map['dataset']], model ,row[column_map['mask_id']]+"_"+DS +"_"+model + EDIT_EXTENSION))
                    return

                perturbed_img_path = os.path.join(PATH_TO_PERTURB_DATA_PARENT , row[column_map['dataset']], model ,row[column_map['mask_id']]+"_"+DS +"_"+model + EDIT_EXTENSION)
                perturbed_img = Image.open(perturbed_img_path)

                if not perturbed_img.getbbox():
                    return

                perturbed_caption = row[column_map['perturbed_label']]

                cap2_img2 = calculate_image_caption_clip_similarity(perturbed_img , perturbed_caption)
                direct_sim = calculate_directional_similarity(orig_img , orig_caption , perturbed_img , perturbed_caption)
                img1_img2  = calculate_image_similarity(orig_img , perturbed_img)
                brisque_score = brisque_Score(perturbed_img)
                row.extend([cap2_img2, direct_sim, img1_img2, brisque_score_orig, brisque_score])
            
                writer.writerow(row)

            except Exception as e:
                
                print(e)
                return
            


def postgen_quality_check(CSV_READ_FILE,
                          CSV_POSTGEN_QC,
                          DS,
                          column_map,
                          PATH_TO_DATA_PARENT,
                          PATH_TO_PERTURB_DATA_PARENT,
                          DS_EXTENSION,
                          EDIT_EXTENSION): 
    f = open(CSV_READ_FILE, 'r')
    file =  csv.reader(f)
    header = next(file)

    header_qc = header
    header_qc.extend(['cap2_img2', 'direct_sim','img1_img2','brisque_score_orig','brisque_score_perturb'])

    out_f = open(CSV_POSTGEN_QC, 'w')
    writer = csv.writer(out_f)
    writer.writerow(header_qc)

    for row in tqdm(file):
        post_qual_check_row(row, 
                           writer, 
                           DS, 
                           column_map, 
                           PATH_TO_DATA_PARENT, 
                           PATH_TO_PERTURB_DATA_PARENT, 
                           DS_EXTENSION, 
                           EDIT_EXTENSION
                           )
        
    f.close()
    out_f.close()

def main():
    DS_EXTENSION = DS_extemsion_dict[DS]
    postgen_quality_check(CSV_READ_FILE,
                          CSV_POSTGEN_QC,
                          DS,
                          column_map,
                          PATH_TO_DATA_PARENT,
                          PATH_TO_PERTURB_DATA_PARENT,
                          DS_EXTENSION,
                          EDIT_EXTENSION)

if __name__ == "__main__":
    main()