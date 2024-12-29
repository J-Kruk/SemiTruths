import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPImageProcessor
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import pdb
import cv2
from brisque import BRISQUE

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_tensor_type("torch.cuda.FloatTensor")

model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID)
model.to(torch_device)

tokenizer = CLIPTokenizer.from_pretrained(model_ID)
preprocess = CLIPImageProcessor.from_pretrained(model_ID)


def preprocess_image(image):
    image = preprocess(image, return_tensors="pt")
    return image


# Define a function to calculate the similarity between two captions
def calculate_caption_clip_similarity(caption1, caption2):
    text_inputs = tokenizer(
        [caption1, caption2],
        padding=True,
        return_tensors="pt",
    ).to(torch_device)
    text_features = model.get_text_features(**text_inputs)
    similarity_features = torch.nn.functional.cosine_similarity(
        text_features[0], text_features[1], dim=-1
    )
    return similarity_features.item()


def prune_length(orig_cap, edit_cap):
    orig_cap = orig_cap.split()
    edit_cap = edit_cap.split()
    if len(orig_cap) + 20 < len(edit_cap):
        return False
    return True


# Define a function to calculate the similarity between an image and a caption
def calculate_image_caption_clip_similarity(image, caption):
    image = preprocess_image(image)["pixel_values"]

    text_inputs = tokenizer(
        [caption],
        padding=True,
        return_tensors="pt",
    ).to(torch_device)

    caption_encoding = model.get_text_features(**text_inputs)
    image_encoding = model.get_image_features(image)
    similarity = torch.nn.functional.cosine_similarity(
        image_encoding, caption_encoding, dim=-1
    )
    return similarity.item()


def calculate_image_similarity(image1, image2):

    image1 = preprocess_image(image1)["pixel_values"]
    image2 = preprocess_image(image2)["pixel_values"]
    image_encoding1 = model.get_image_features(image1)
    image_encoding2 = model.get_image_features(image2)
    similarity = torch.nn.functional.cosine_similarity(
        image_encoding1, image_encoding2, dim=-1
    )
    return similarity.item()


# Define a function to calculate the similarity between an image and a caption
def calculate_directional_similarity(image1, caption1, image2, caption2):
    image1 = preprocess_image(image1)["pixel_values"]
    image2 = preprocess_image(image2)["pixel_values"]

    text_inputs = tokenizer(
        [caption1, caption2],
        padding=True,
        return_tensors="pt",
    ).to(torch_device)

    caption_encoding = model.get_text_features(**text_inputs)
    image_encoding1 = model.get_image_features(image1)
    image_encoding2 = model.get_image_features(image2)
    similarity = torch.nn.functional.cosine_similarity(
        image_encoding1 - caption_encoding[0],
        image_encoding2 - caption_encoding[1],
        dim=-1,
    )

    return similarity.item()


def brisque_Score(img):
    ndarray = np.asarray(img)
    obj = BRISQUE(url=False)
    score = obj.score(img=ndarray)
    return score
