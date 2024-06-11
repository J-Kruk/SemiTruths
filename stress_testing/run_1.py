import argparse
import glob
import os
import sys
from functools import lru_cache
import torch
import csv
import pandas as pd
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from tqdm import tqdm
import random
import clip
import pdb
import subprocess

sys.path.append("../DeFake/.")
from DeFake.test import NeuralNet
from DeFake.blipmodels.blip import blip_decoder

# from gradcam import get_gradcam_ops

sys.path.append("../DIRE/.")
from DIRE.utils.utils import get_network, str2bool, to_cuda
from CNNDetection.networks.resnet import resnet50
from UniversalFakeDetect.models import get_model

sys.path.append("../")
import preprocessing.utils as utils
import numpy as np
import pdb
import importlib.util

import torchvision.models as models
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-f",
    "--file",
    default="/data/jkruk3/half-truths/data_samples/eval_sample_5p/full/data_sample_5p_flat.csv",
    type=str,
    help="path to csv of image metadata",
)
parser.add_argument(
    "-r",
    "--root_dir",
    default="/data/jkruk3/half-truths/data_samples/eval_sample_5p/full/",
    type=str,
    help="path to root directory of image metadata",
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="CNNDetection/weights/blur_jpg_prob0.1.pth",
)
parser.add_argument("--arch", type=str, default="resnet50")
parser.add_argument(
    "--ckpt",
    type=str,
    default="./UniversalFakeDetect/pretrained_weights/fc_weights.pth",
    help="Path to model checkpoint",
)
# parser.add_argument("--aug_norm", type=str2bool, default=True)
parser.add_argument(
    "--model",
    type=str,
    default="dire",
    help="options: dire, cnnspot, defake, universalfakedetect, dinov2, crossefficientvit",
)
parser.add_argument(
    "--crossefficient_output_dir",
    default="/data/jkruk3/half-truths/CrossEfficient_Interm_Data/sample_5p",
    type=str,
    help="Output directory for interm media.",
)
parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("-j", "--workers", type=int, default=4, help="Number of workers")
parser.add_argument(
    "--size_only", action="store_true", help="Only look at sizes of images in dataset"
)
parser.add_argument("--save_path", type=str, default="dire_preds.csv")
parser.add_argument("--output_path", type=str, default="gradcam_op")
parser.add_argument("--gradcam_method", type=str, default="gradcam")
parser.add_argument(
    "--aug-smooth",
    action="store_true",
    help="Apply test time augmentation to smooth the CAM",
)
parser.add_argument(
    "--eigen-smooth",
    action="store_true",
    help="Reduce noise by taking the first principle component"
    "of cam_weights*activations",
)

args = parser.parse_args()
file_type_map = {
    "ADE20K": ".jpg",
    "CelebAHQ": ".jpg",
    "CityScapes": ".png",
    "HumanParsing": ".png",
    "OpenImages": ".jpg",
    "SUN_RGBD": ".jpg",
}

oserr_count = 0

if os.path.isfile(args.file):
    print(f"Testing on file '{args.file}'")
    df = pd.read_csv(args.file)
    df["prediction"] = [pd.NA] * len(df)
    df["probability"] = [pd.NA] * len(df)

trans = transforms.Compose(
    (
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


######## De-Fake functions ################
@lru_cache(maxsize=128)
def import_NN():
    module_name = "De-Fake.test"
    module_path = os.path.join(os.path.dirname(__file__), "De-Fake", "test.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.NeuralNet


@lru_cache(maxsize=128)
def import_blip_decoder():
    module_name = "De-Fake.models.blip"
    module_path = os.path.join(
        os.path.dirname(__file__), "De-Fake", "models", "blip.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.blip_decoder


####### DINOV2 functions #################


def load_model():
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    return model, processor


def load_fake_image_detector():
    # Load the pre-trained ResNet model for fake image detection
    resnet_model = models.resnet50(pretrained=True)
    # Freeze all the parameters in the ResNet model
    for param in resnet_model.parameters():
        param.requires_grad = False
    # Modify the last fully connected layer to output 1 (binary classification)
    num_features = resnet_model.fc.in_features
    resnet_model.fc = torch.nn.Linear(num_features, 1)
    return resnet_model


def predict_image_dino(image, dino_model, processor, fake_detector_model):
    # Predict if an image is real or fake using DinoV2 and a pre-trained ResNet model.
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        dino_outputs = dino_model(**inputs)
    dino_logits = dino_outputs.last_hidden_state

    # Assuming DINO's last layer hidden states are used as features
    dino_features = dino_logits[:, 0]

    # Reshape the features tensor to [batch_size, channels, height, width]
    batch_size = dino_features.shape[0]
    dino_features = dino_features.view(batch_size, 1, 1, -1)

    # Resize the features tensor to match the input size expected by the pre-trained ResNet model
    dino_features_resized = F.interpolate(
        dino_features, size=(224, 224), mode="bilinear", align_corners=False
    )

    # Convert the features tensor to have three channels (RGB)
    dino_features_rgb = dino_features_resized.expand(-1, 3, -1, -1)

    # Make predictions with the pre-trained ResNet model
    fake_detector_model.eval()
    with torch.no_grad():
        predictions = torch.sigmoid(fake_detector_model(dino_features_rgb))

    return predictions


########## UniversalFakeDetect functions #################


@lru_cache(maxsize=128)
def predict_image_ufd(in_tens, model):
    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            in_tens = in_tens.cuda()
        if in_tens.shape[2] > 224:
            print("image too large")
            pdb.set_trace()
        pred = torch.sigmoid(model(in_tens))
        prob = pred.item()
    return pred.item()


########################################

if args.model == "cnnspot":
    print("CNNSpot")
    model = resnet50(num_classes=1)
    if args.model_path is not None:
        state_dict = torch.load(args.model_path, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

elif args.model == "dire":
    print("DIRE")
    model = get_network(args.arch)
    if args.model_path is not None:
        state_dict = torch.load(args.model_path, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

elif args.model == "defake":
    print("De-Fake")
    model = torch.load("DeFake/finetune_clip.pt").to(device)
    linear = NeuralNet(1024, [512, 256], 2).to(device)
    linear = torch.load("DeFake/clip_linear.pt")

    model2, preprocess = clip.load("ViT-B/32")

    image_size = 224

    blip_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"

    blip = blip_decoder(
        pretrained=blip_url,
        med_config="DeFake/blipmodels/blipconfig/med_config.json",
        image_size=image_size,
        vit="base",
    )
    blip.eval()
    blip = blip.to(device)
    model.to(device)

elif args.model == "dinov2":
    print("DinoV2 with pretrained ResNet")
    dino_model, processor = load_model()
    detector_model = load_fake_image_detector()
    dino_model = dino_model.to(device)
    detector_model = detector_model.to(device)
    # processor = processor.to(device)

elif args.model == "universalfakedetect":
    print("UniversalFakeDetect")
    model = get_model("CLIP:ViT-L/14")
    state_dict = torch.load(args.ckpt)  # Load on CUDA if available
    model.fc.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.cuda()

elif args.model == "crossefficientvit":
    print("CrossEfficientViT")

if args.model != "crossefficientvit":
    mask_trans_funt = transforms.Compose(
        (transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor())
    )

    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(args.root_dir, row.image_path)
        img_id = row.image_id
        mask_path = row.mask_path

        # NOTE: masks exists, therefore fake image
        if not pd.isna(mask_path):
            mask_path = os.path.join(args.root_dir, mask_path)

            # ensuring proper file name:
            if (
                ".png" not in img_path
                and ".jpg" not in img_path
                and ".jpeg" not in img_path
            ):
                dataset = [d for d in file_type_map.keys() if d in img_path][0]
                file_type = file_type_map[dataset]
                img_path = img_path + file_type

            if os.path.exists(img_path):

                output_path = os.path.join(
                    args.output_path,
                    img_path.split("/")[-3],
                    img_path.split("/")[-2],
                )
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                try:
                    mask_orig = Image.open(mask_path).convert("RGB")
                except FileNotFoundError:
                    mask_path = mask_path.replace(".jpg", ".png")
                    mask_orig = Image.open(mask_path).convert("RGB")
                    continue
                trans_resize = transforms.Compose(
                    (transforms.Resize(256), transforms.ToTensor())
                )
                mask_trans = (mask_trans_funt(mask_orig)).to(device)
                mask_orig = (trans_resize(mask_orig)).to(device)

                good_crop = False
                ratio = 0
                _, height, width = mask_orig.size()
                center_pixel = (height // 2, width // 2)
                max_mask_ratio = 0
                tries = 0
                center_pixel_max = center_pixel
                while good_crop != True:

                    # mask_orig.save("test_mask_for_ratio_OLD.png")
                    # mask_trans.save("test_mask_for_ratio_NEW.png")
                    r, mask_exists = utils.verify_mask(mask_trans)

                    if mask_exists:
                        mask_sa_old = utils.mask_sa_ratio(mask_orig)
                        mask_sa_new = utils.mask_sa_ratio(mask_trans)
                        try:
                            mask_ratio = mask_sa_new / mask_sa_old
                        except:
                            img = Image.open(img_path).convert("RGB")
                            img.save("img_w_empty mask.png")

                        if mask_ratio > max_mask_ratio:
                            max_mask_ratio = mask_ratio
                            center_pixel_max = center_pixel

                        if mask_ratio >= 0.7 or mask_ratio == 0.0 or tries > 50:
                            ratio = r
                            good_crop = True
                            img = Image.open(img_path).convert("RGB")
                            img_trans = trans_resize(img)
                            if tries > 50:
                                center_pixel = center_pixel_max
                            img_trans = transforms.functional.crop(
                                img_trans,
                                center_pixel[0],
                                center_pixel[1],
                                224,
                                224,
                            )
                            trans_new = transforms.Compose(
                                (
                                    # transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225],
                                    ),
                                )
                            )

                            img_trans = trans_new(img_trans)
                            in_tens = (
                                torch.clamp(img_trans, min=0.0, max=1.0)
                                .unsqueeze(0)
                                .to(device)
                            )
                            # break

                        else:
                            tries += 1
                            # Recrop randomly and save the center pixel of this crop
                            crop_center = (
                                random.randint(112, mask_orig.size()[1] - 112),
                                random.randint(112, mask_orig.size()[2] - 112),
                            )
                            # print(crop_center)
                            crop_position = (
                                crop_center[0] - 112,
                                crop_center[1] - 112,
                            )
                            mask_trans = transforms.functional.crop(
                                mask_orig,
                                crop_position[0],
                                crop_position[1],
                                224,
                                224,
                            )
                            center_pixel = crop_center
                    else:
                        if (
                            ".png" not in img_path
                            and ".jpg" not in img_path
                            and ".jpeg" not in img_path
                        ):
                            dataset = [
                                d for d in file_type_map.keys() if d in img_path
                            ][0]
                            file_type = file_type_map[dataset]
                            img_path = img_path + file_type

                        if os.path.exists(img_path):
                            img = Image.open(img_path).convert("RGB")
                            if args.model == "universalfakedetect":
                                img = img.filter(
                                    ImageFilter.GaussianBlur(radius=2)
                                )  # Apply Gaussian blur
                            img = trans(img)

                            assert img.shape[2] <= 225
                            in_tens = (
                                torch.clamp(img, min=0.0, max=1.0)
                                .unsqueeze(0)
                                .to(device)
                            )
                            break

                        else:
                            print(img_path)
                            print("Cannot find file")
                            pdb.set_trace()
                            df.at[i, "prediction"] = pd.NA
                            df.at[i, "probability"] = pd.NA
                            continue

            else:
                df.at[i, "prediction"] = pd.NA
                df.at[i, "probability"] = pd.NA
                continue

        # NOTE: masks does not exist, therefore real image
        else:
            # ensuring proper file name:
            if (
                ".png" not in img_path
                and ".jpg" not in img_path
                and ".jpeg" not in img_path
            ):
                dataset = [d for d in file_type_map.keys() if d in img_path][0]
                file_type = file_type_map[dataset]
                img_path = img_path + file_type

            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")

                if args.model == "universalfakedetect":
                    img = img.filter(
                        ImageFilter.GaussianBlur(radius=2)
                    )  # Apply Gaussian blur
                img = trans(img)

                # in_tens = img.unsqueeze(0).to(device)
                assert img.shape[1] <= 225
                in_tens = torch.clamp(img, min=0.0, max=1.0).unsqueeze(0).to(device)

            else:
                print(img_path)
                print("Cannot find file")
                pdb.set_trace()
                df.at[i, "prediction"] = pd.NA
                df.at[i, "probability"] = pd.NA
                continue

        if args.model == "cnnspot":
            # print("Continuing CNNSpot")
            with torch.no_grad():
                prob = model(in_tens).sigmoid().item()
            pred = "1" if prob >= 0.50 else "0"

        elif args.model == "dire":
            # print("Continuing DIRE")
            with torch.no_grad():
                prob = model(in_tens).sigmoid().item()
            pred = "1" if prob >= 0.50 else "0"

        elif args.model == "defake":
            # print("Continuing De-Fake")
            # try:
            caption = blip.generate(
                in_tens, sample=False, num_beams=3, max_length=60, min_length=5
            )
            # except:
            #     pdb.set_trace()
            # print(caption)
            text = clip.tokenize(list(caption)).to(device)

            with torch.no_grad():
                image_features = model.encode_image(in_tens)
                text_features = model.encode_text(text)

                emb = torch.cat((image_features, text_features), 1)
                output = linear(emb.float())
                pred = output.argmax(1)
                pred = pred.cpu().numpy()
                pred = pred[0]

                prob = torch.sigmoid(output.squeeze()[pred])
                prob = float(prob.cpu().numpy())
                if pred == 0:
                    prob = 1 - prob

        elif args.model == "dinov2":
            # print("Continuing DinoV2")
            prob = predict_image_dino(
                in_tens, dino_model, processor, detector_model
            ).item()
            pred = "1" if prob >= 0.50 else "0"

        elif args.model == "universalfakedetect":
            prob = predict_image_ufd(in_tens, model)
            pred = "1" if prob >= 0.50 else "0"

        df.at[i, "probability"] = round(prob, 2)
        df.at[i, "prediction"] = pred

    print("predictions done, creating csv")
    if not os.path.exists("preds"):
        os.makedirs("preds")

    df.to_csv(os.path.join("preds", (args.model + "_preds.csv")), index=False)

elif args.model == "crossefficientvit":
    file = "Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/preprocessing/detect_faces.py"
    facedetect = ["python3", file, "--data_path", args.root_dir]
    subprocess.run(facedetect)

    # file = "Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/preprocessing/extract_crops.py"
    # extractcrop = [
    #     "python3",
    #     file,
    #     "--metadata",
    #     args.file,
    #     "--data_path",
    #     args.root_dir,
    #     "--output_dir",
    #     args.crossefficient_output_dir,
    # ]
    # subprocess.run(extractcrop)

    file = "Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/cross-efficient-vit/test_imgs.py"
    test = [
        "python3",
        file,
        "--metadata",
        args.file,
        "--model_path",
        "Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/cross-efficient-vit/cross_efficient_vit.pth",
        "--config",
        "Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/cross-efficient-vit/configs/architecture.yaml",
        "--output_dir",
        "preds/",
        "--media_dir_root",
        args.crossefficient_output_dir,
    ]
    subprocess.run(test)
