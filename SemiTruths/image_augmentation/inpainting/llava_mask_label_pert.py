import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
sys.path.append("../")
sys.path.append("../LLaVA")

from tqdm import tqdm
from SemiTruths.image_augmentation.inpainting.run_llava import (
    eval_model,
    load_model,
    eval_model_wo_loading,
)
import pdb
import diffusers
import transformers

print("diffusers version:  ", diffusers.__version__)
print("transformers version:  ", transformers.__version__)


# from LLaVA.run_llava import eval_model, load_model, eval_model_wo_loading
from diffusers import (
    StableDiffusionInpaintPipeline,
    AutoPipelineForInpainting,
    DiffusionPipeline,
    IFInpaintingSuperResolutionPipeline,
)
from .utils import load_data_flat, InpaintingPertDataset, DataLoader, get_llava_perts
from sentence_transformers import SentenceTransformer, util
from accelerate import Accelerator
from datasets import load_dataset
import json
import torch
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pdb
from huggingface_hub import login
import warnings
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
torch.set_warn_always(False)
warnings.filterwarnings("ignore")

## HuggingFace Log-in ##
# login(token="")


def perturb_mask_labels(args):
    """
    Uses LLaVA to perturb entity mask labels to reflect various
    magnitudes of changes.
    """
    if not os.path.exists(args.output_dir_pert):
        os.mkdir(args.output_dir_pert)
    OUTPATH = os.path.join(args.output_dir_pert, "metadata_pert.json")
    VIZ_DIR = os.getcwd()

    ## DATA LOADING ##
    metadata = load_data_flat(args.input_data)
    processed_metadata = {}
    print(f"Number of images for mask label perturbation: {len(metadata)}\n")

    data_ds = InpaintingPertDataset(metadata)
    inf_dataloader = DataLoader(data_ds, batch_size=1, shuffle=True)

    ## MODEL LOADING  ##
    model_path = args.llava_model
    prompt = ""
    image_file = ""
    args_llava = type(
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
            "cache_dir": args.llava_cache_dir,
        },
    )()

    accelerator = Accelerator()
    tokenizer, model, image_processor, context_len = load_model(args_llava)
    model, inf_dataloader, tokenizer, image_processor = accelerator.prepare(
        model, inf_dataloader, tokenizer, image_processor
    )

    ## PROMPT STRUCTURING ##
    context = """
    Semantic change in images refers to alterations in meaning or interpretation. Here's how you might describe small, medium, and large semantic changes to an image:

    1. Small Semantic Change: Would not significantly alter the overall meaning or interpretation of the image. Changes to lighting, an object's color, adding or removing a minor detail, and minor alterations to the background are considered small semantic changes.
    2. Medium Semantic Change: Could slightly alter the way the viewer perceives the image and its subject matter. Changes to attributes of the subject or objects in the foreground, the orinetation of an object or person, the emotions of the people in the frame, and noicable alterations to background elements are considered medium semantic changes.
    3. Large Semantic Change: Involves substantial modifications that fundamentally transform the interpretation or message conveyed by the image. It may even appear suprising or strange to an audience. Changes that alter, add or remove key elements of the image, such as people, objects, and landmarks, replace the subject of the image, and dramatic alteration to the color scheme or mood are considered large semantic changes. 

    Using the definitions of small, medium, and large semantic changes above, """

    ## PERTURBING ##
    processed = get_llava_perts(
        metadata,
        processed_metadata,
        args.input_data,
        OUTPATH,
        context,
        args_llava,
        tokenizer,
        model,
        image_processor,
        context_len,
        VIZ_DIR,
        retry=3,
    )


if __name__ == "__main__":

    ## ARGPARSE ##
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_data",
        default="../../data/input",
        help="Path to input media.",
    )
    parser.add_argument(
        "--metadata",
        default="../../data/input/metadata.csv",
        help="Path to input metadata.",
    )
    parser.add_argument(
        "--llava_model",
        default="liuhaotian/llava-v1.6-mistral-7b",
        help="LLaVA-Mistral huggingface model id.",
    )
    parser.add_argument(
        "--llava_cache_dir",
        default="../llava_cache",
        help="Directory to store LLaVA cache files.",
    )
    parser.add_argument(
        "--output_dir_pert",
        default="../../data",
        help="Directory to save metadata with perturbed mask labels.",
    )
    args = parser.parse_args()

    perturb_mask_labels(args)
    print("LLaVA-Mistral mask label perturbations COMPLETE!!!")
