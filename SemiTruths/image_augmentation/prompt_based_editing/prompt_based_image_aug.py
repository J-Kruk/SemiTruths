# Copyright 2023 the LANCE team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

import os
import pdb
import sys
import json
import logging
import cv2
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torchvision.datasets as datasets
from accelerate import Accelerator
import pkgutil

# sys.path.append("../../../SemiTruths")
from accelerate.logging import get_logger
from LANCE.lance.generate_captions import *
from LANCE.lance.edit_captions import *
from LANCE.lance.edit_images import *
from LANCE.lance.utils.misc_utils import *

accelerator = Accelerator()


def configure_lance_args(parser, semi_args, diff_model, dataset):
    """
    To make LANCE pipeline arguments align with those
    within the SemiTruths code base, we dynamically
    create new arguement that LANCE code expects at input.
    """
    parser.add_argument(
        "--ldm_type",
        type=str,
        default=semi_args.diff_model,
        help="Latent Diffusion Model to use",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=semi_args.input_data_pth,
        help="ImageFolder containing images",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=semi_args.input_metadata_pth,
        help="Path to input metadata json",
    )
    parser.add_argument(
        "--ldm_type",
        type=str,
        default=diff_model,
        help="Path to input metadata json",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=dataset,
        help="Dataset to parse",
    )

    args = parser.parse_args()
    return args


def setup_data(args: argparse.Namespace, logger):
    if args.dset_name == "HardImageNet":
        import LANCE.datasets.hard_imagenet as ha

        dset = ha.HardImageNet(args.img_dir)
        if args.verbose:
            logger.info(f"=> Loaded dataset from {args.img_dir}")

    elif args.dset_name == "ImageFolder":
        import LANCE.datasets.custom_imagefolder as cif

        dset = cif.CustomImageFolder(args.img_dir, args.json_path, args.dataset)
        if args.verbose:
            logger.info(f"=> Loaded dataset from {args.img_dir}")
    else:
        logger.error("Dataset type not supported, exiting")
        raise ValueError("Dataset not supported")

    data_sampler = torch.utils.data.sampler.SequentialSampler(dset)
    dataloader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        shuffle=(data_sampler is None),
        num_workers=6,
        pin_memory=True,
        sampler=data_sampler,
        drop_last=True,
    )


def initialize_dicts(args: argparse.Namespace, logger):
    gencap_dict = {}
    if args.load_captions:
        if not os.path.exists(args.gencap_dict_path):
            logger.error("Path to caption file does not exist")
            raise ValueError
        gencap_dict = json.load(open(args.gencap_dict_path, "r"))
        if args.verbose:
            logger.info(f"=> Loaded generated captions from {args.gencap_dict_path}")

    editcap_dict = {}
    if args.load_caption_edits:
        if not os.path.exists(args.editcap_dict_path):
            raise ValueError("Path to edited caption file does not exist")
        editcap_dict = json.load(open(args.editcap_dict_path, "r"))
        if args.verbose:
            logger.info(f"=> Loaded edited captions from {args.editcap_dict_path}")
    return gencap_dict, editcap_dict


def prompt_based_edit_modules(args: argparse.Namespace, logger, device):
    if args.verbose:
        logger.info(f"=> Initializing image editor")
    image_editor = ImageEditor(
        args,
        device,
        verbose=args.verbose,
        similarity_metric=ClipSimilarity(device=device),
        text_similarity_threshold=args.text_similarity_threshold,
        ldm_type=args.ldm_type,
        save_inversion=args.save_inversion,
        edit_word_weight=args.edit_word_weight,
        clip_thresh=args.clip_thresh,
        clip_img_thresh=args.clip_img_thresh,
        clip_dir_thresh=args.clip_dir_thresh,
    )

    if not args.load_captions:
        if args.verbose:
            logger.info(f"=> Initializing image captioner")
        caption_generator = CaptionGenerator(
            args,
            device,
            verbose=args.verbose,
            name=args.gencaption_name,
        )
    else:
        caption_generator = None

    if not args.load_caption_edits:
        if args.verbose:
            logger.info(f"=> Initializing caption editor")
        caption_editor = CaptionEditor(
            args, device, verbose=args.verbose, perturbation_type=args.perturbation_type
        )
    else:
        caption_editor = None
    return image_editor, caption_generator, caption_editor


def generate_edits(
    args: argparse.Namespace,
    logger,
    dataloader,
    image_editor,
    caption_generator,
    caption_editor,
    gencap_dict,
    editcap_dict,
):

    for paths, targets in tqdm(dataloader, total=len(dataloader)):
        # Generate caption
        img_path, clsname = paths[0], targets[0]
        if len(np.array(Image.open(img_path)).shape) < 3:
            continue  # Ignore grayscale images
        out_dir = os.path.join(
            args.lance_output_path, "prompt-based-editing", args.ldm_type
        )
        os.makedirs(out_dir, exist_ok=True)

        if args.verbose:
            logger.info(f"=>Generating LANCE for {img_path}")

        img_name = img_path.split("/")[-1]
        if img_name in gencap_dict.keys():
            if args.verbose:
                logger.warning("Caption already generated, loading from dictionary\n")
            cap = gencap_dict[img_name]

        else:
            cap = caption_generator.generate(img_path)
            gencap_dict[img_name] = cap

        # Edit caption
        if img_name in editcap_dict.keys():
            if args.verbose:
                logger.warning(
                    "Caption edits already generated, loading from dictionary\n"
                )
            new_caps = editcap_dict[img_name]

        else:
            new_caps = caption_editor.edit(
                cap, perturbation_type=args.perturbation_type
            )
            editcap_dict[img_name] = new_caps

        out_path = os.path.join(out_dir, os.path.splitext(img_name)[0])
        if os.path.exists(out_path):
            files_total = [
                item
                for item in os.listdir(out_path)
                if os.path.isfile(os.path.join(out_path, item))
            ]
            if len(files_total) > 4:
                if args.verbose:
                    logger.warning(f"=> Image `{out_path}' already edited, skipping")
                continue

        # Invert image
        _, _, x_t, uncond_embeddings = image_editor.invert(img_path, cap, out_dir)

        # Edit image
        image_editor.edit(
            out_path, clsname.lower(), x_t, uncond_embeddings, cap, new_caps
        )
        del x_t, uncond_embeddings
        accelerator.free_memory()
    return gencap_dict, editcap_dict, args


def save_jsons(gencap_dict, editcap_dict, args):
    save_path_edited = os.path.join(
        out_dir, "edited_captions", args.json_path.split("/")[-1]
    )
    json.dump(gencap_dict, open(args.gencap_dict_path, "w"), indent=4)
    json.dump(editcap_dict, open(args.editcap_dict_path, "w"), indent=4)
    json.dump(vars(args), open(args.lance_output_path + "/args.json", "w"), indent=4)


def create_prompt_based_edits(args: argparse.Namespace):

    logging.info(accelerator.state, main_process_only=True)
    device = accelerator.device

    if args.verbose:
        logger = get_logger("lance")
        for arg, value in sorted(vars(args).items()):
            logger.debug("{}: {}", arg, value)
        logger.info("------------------------------------------------")
        logger.info(f"=> Initializing LANCE")

    dataloader = setup_data(args, logger)
    gencap_dict, editcap_dict = initialize_dicts(args, logger)

    image_editor, caption_generator, caption_editor = prompt_based_edit_modules(
        args, logger, device
    )

    model = image_editor.model
    dataloader, model = accelerator.prepare(dataloader, model)
    gencap_dict, editcap_dict, args = generate_edits(
        args,
        logger,
        dataloader,
        image_editor,
        caption_generator,
        caption_editor,
        gencap_dict,
        editcap_dict,
    )

    save_jsons(gencap_dict, editcap_dict, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ###########################################################################
    # Experiment identifier
    ###########################################################################

    parser.add_argument(
        "--dataset_type",
        type=str,
        help="Dataset type: HardImageNet or ImageFolder",
        default="ImageFolder",
    )
    parser.add_argument("--img_dir", type=str, help="ImageFolder containing images")
    parser.add_argument(
        "--json_path", type=str, default="outputs", help="Path to input metadata json"
    )
    parser.add_argument(
        "--lance_output_path",
        type=str,
        default="outputs",
        help="LANCE output directory",
    )

    ###########################################################################
    # Caption generator hyperparameters
    ###########################################################################

    parser.add_argument(
        "--gencaption_name",
        type=str,
        help="Captioner name: blip2_opt,blip2_t5,blip2,blip_caption",
        default="blip_caption",
    )
    parser.add_argument(
        "--load_captions", action="store_true", help="Load captions from path"
    )
    parser.add_argument(
        "--gencap_dict_path",
        type=str,
        default="outputs/hard_imagenet_captions_blip2.json",
        help="Path to JSON file containing image captions",
    )
    parser.add_argument(
        "--load_caption_edits", action="store_true", help="Load captions from path"
    )
    parser.add_argument(
        "--editcap_dict_path",
        type=str,
        default="outputs/hard_imagenet_captions_blip2_edited.json",
        help="Path to JSON file containing edited captions",
    )

    ###########################################################################
    # Caption editor hyperparameters
    ###########################################################################

    parser.add_argument(
        "--llama_finetuned_path",
        type=str,
        # default="LANCE/checkpoints/caption_editing/lit-llama-lora-finetuned.pth",
        default="./LANCE/checkpoints/caption_editing/lit-llama-lora-finetuned.pth",
        help="Path to finetuning llama model in lightning format",
    )
    parser.add_argument(
        "--llama_pretrained_path",
        type=str,
        # default="LANCE/checkpoints/caption_editing/lit-llama-lora-finetuned.pth",
        default="/data/jkruk3/half-truths/lance_checkpoints/lit-llama.pth",
        help="Path to pretrained llama model in lightning format",
    )
    parser.add_argument(
        "--llama_tokenizer_path",
        type=str,
        default="./LANCE/checkpoints/caption_editing/tokenizer.model",
        help="Path to LLAMA tokenizer model",
    )
    parser.add_argument(
        "--perturbation_type",
        type=str,
        default="all",
        help="Type of perturbation to stress-test against",
    )

    ###########################################################################
    # Image editing hyperparameters
    ###########################################################################

    parser.add_argument(
        "--ldm_type",
        type=str,
        default="stable_diffusion_v1_4",
        help="Latent Diffusion Model to use",
    )
    parser.add_argument(
        "--text_similarity_threshold",
        type=float,
        default=1.0,
        help="Threshold for CLIP text similarity between GT class and word(s) being edited",
    )
    parser.add_argument(
        "--clip_img_thresh",
        type=float,
        default=0.0,
        help="Threshold for CLIP similarity between original and edited image",
    )
    parser.add_argument(
        "--clip_dir_thresh",
        type=float,
        default=0.0,
        help="Threshold for CLIP similarity between original and edited direction",
    )
    parser.add_argument(
        "--clip_thresh",
        type=float,
        default=0.0,
        help="Threshold for CLIP similarity between original and edited image and direction",
    )
    parser.add_argument(
        "--edit_word_weight",
        type=float,
        default=2.0,
        help="Maximum number of tries for editing a caption",
    )
    parser.add_argument(
        "--save_inversion",
        action="store_false",
        help="Whether to save image inversion and load from it for future edits",
    )
    parser.add_argument(
        "--verbose",
        action="store_false",
        help="Logging verbosity",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ADE20K",
        help="Dataset to parse",
    )

    ###########################################################################

    args = parser.parse_args()
    create_prompt_based_edits(args)
