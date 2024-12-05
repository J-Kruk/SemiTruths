cd image_augmentation/inpainting/
python llava_mask_label_pert.py --input_data ../../data/input --metadata ../../data/input/metadata.csv --llava_model liuhaotian/llava-v1.6-mistral-7b --llava_cache_dir ../llava_cache --output_dir ../../data
python llava_guided_inpainting.py --diff_model StableDiffusion_v4 --input_data ../../data/input --output_dir ../../data/gen --pert_file ../../data/metadata_pert.json
cd ../LANCE/
python main.py --dset_name "ImageFolder"  --img_dir '../../../data/input' --json_path ../../../data/input/metadata.csv  --ldm_type 'SDv4' --lance_path ../../../data --dataset 'ADE20K' --editcap_dict_path ../../../data/editcap_dict.json --gencap_dict_path ../../../data/prompt-prompt/gencap_dict.json

