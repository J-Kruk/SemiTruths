import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import sys
from tqdm import tqdm
from run_llava import eval_model, load_model, eval_model_wo_loading
from diffusers import (
    StableDiffusionInpaintPipeline,
    AutoPipelineForInpainting,
    DiffusionPipeline,
    IFInpaintingSuperResolutionPipeline,
)
from utils import *
from sentence_transformers import SentenceTransformer, util
from accelerate import Accelerator
from datasets import load_dataset
import PIL
from PIL import Image, ImageFilter
import random
import json
import requests
import torch
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import ipdb
from huggingface_hub import login
import warnings

login(token="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")
torch.set_warn_always(False)
warnings.filterwarnings("ignore")


## DATA LOADING ##

CACHE_DIR = "/home/jkruk3/how_fake/inpainting/LLaVA/cache"
DATA_DIR = "/data/jkruk3/Half_Truths_Dataset"
ROOT_DIR = "/data/jkruk3"
OUTPATH = os.path.join(DATA_DIR, "metadata_pert_ade20k.json")
VIZ_DIR = os.getcwd()
# MODEL = "liuhaotian/llava-v1.6-mistral-7b"
MODEL = "liuhaotian/llava-v1.6-34b"
DS = "ADE20K"  # benchmark to sample data from for this run

metadata, processed_metadata = load_data(DATA_DIR, OUTPATH)
# metadata, processed_metadata = load_data_hf(
#     "Half-Truths-Project/base-datasets-2", DATA_DIR, OUTPATH
# )

data_samp = sample_by_dataset(metadata, DS)
print(f"Number of images for mask label perturbation: {len(data_samp)}\n")
print(f"Number of images for already processed: {len(processed_metadata)}\n")

data_ds = InpaintingPertDataset(data_samp)
inf_dataloader = DataLoader(data_ds, batch_size=1, shuffle=True)

## MODEL LOADING  ##

# LLaVA model agruments for LLaVA-Mistral:
model_path = MODEL
prompt = ""
image_file = ""
args = type(
    "Args",
    (),
    {
        "model_path": model_path,
        "model_base": None,
        "model_name": "LlavaLlamaForCausalLM",
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0.5,
        "top_p": 0.5,
        "num_beams": 5,
        "max_new_tokens": 512,
        "cache_dir": CACHE_DIR,
    },
)()

accelerator = Accelerator()
tokenizer, model, image_processor, context_len = load_model(args)
model, inf_dataloader, tokenizer, image_processor = accelerator.prepare(
    model, inf_dataloader, tokenizer, image_processor
)

## PROMPT STRUCTURING ##

context_v0 = """
Semantic change in images refers to alterations in meaning or interpretation. Here's how you might describe small, medium, and large semantic changes to an image:

1. Small Semantic Change: Could be something like changing the color of a specific object, adding or removing a minor detail, adjusting the composition or perspective of the image, or slightly adjusting the color distribution of the image. This change would not significantly alter the overall meaning or context of the image.
2. Medium Semantic Change: Could involve minor changes to attributes of an object or setting, alterating a background element, moving an object or person to a different location within the frame, or changing the emotions of the people in the frame. This change could slightly alter the way the viewer perceives the image and its subject matter.
3. Large Semantic Change: This could include altering, adding or removing major elements of the image, such as people, objects, and landmarks, or making fundamental changes to the image's color scheme, lighting, or other visual elements. Involves substantial modifications that fundamentally transform the interpretation or message conveyed by the image. It may even appear suprising or strange to an audience.

Using the definitions of small, medium, and large semantic changes above, """

context = """
Semantic change in images refers to alterations in meaning or interpretation. Here's how you might describe small, medium, and large semantic changes to an image:

1. Small Semantic Change: Would not significantly alter the overall meaning or interpretation of the image. Changes to lighting, an object's color, adding or removing a minor detail, and minor alterations to the background are considered small semantic changes.
2. Medium Semantic Change: Could slightly alter the way the viewer perceives the image and its subject matter. Changes to attributes of the subject or objects in the foreground, the orinetation of an object or person, the emotions of the people in the frame, and noicable alterations to background elements are considered medium semantic changes.
3. Large Semantic Change: Involves substantial modifications that fundamentally transform the interpretation or message conveyed by the image. It may even appear suprising or strange to an audience. Changes that alter, add or remove key elements of the image, such as people, objects, and landmarks, replace the subject of the image, and dramatic alteration to the color scheme or mood are considered large semantic changes. 

Using the definitions of small, medium, and large semantic changes above, """


processed = get_llava_perts(
    data_samp,
    processed_metadata,
    ROOT_DIR,
    OUTPATH,
    context,
    args,
    tokenizer,
    model,
    image_processor,
    context_len,
    VIZ_DIR,
    retry=3,
)

ipdb.set_trace()
