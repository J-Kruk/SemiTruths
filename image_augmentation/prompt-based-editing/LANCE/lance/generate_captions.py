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
import time
import pdb
import argparse
from typing import Optional
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger

from lance.utils.misc_utils import *


class CaptionGenerator:
    def __init__(
        self,
        args: argparse.Namespace,
        device: torch.device,
        verbose: bool = False,
        name: Optional[str] = "blip_caption",
        model_type: Optional[str] = "large_coco",
        repetition_penalty=1.0,
        min_caption_length=20,
        max_caption_length=100,
    ):
        """
        Initialize caption generator
        Args:
            args: Command line arguments from argparse
            device: Device to run model on
            name: Name of model. Defaults to "blip_caption".
            model_type: Type of model. Defaults to "large_coco".
            repetition_penalty: Repetition penalty. Defaults to 1.0.
        """
        self.args = args
        self.repetition_penalty = repetition_penalty
        self.min_length = min_caption_length
        self.max_length = max_caption_length
        self.device = device
        t0 = time.time()
        self.verbose = verbose
        if self.verbose:
            logger.info("[Loading BLIP-2 model]")
        if(name == 'blip2_opt'):
            model_type = 'caption_coco_opt6.7b'
        elif(name =='blip2_t5'):
            model_type='caption_coco_flant5xl'
        elif(name =='blip2'):
            model_type = 'coco'
        elif(name=='blip_caption'):
            model_type ='large_coco'
            
        self.model, self.vis_processors, _ = load_model_and_preprocess(name=name, model_type=model_type, is_eval=True, device=self.device)
        if self.verbose:
            logger.debug(f"Time to load model: {time.time() - t0:.02f} seconds.")
        self.vis_processors.keys()

    def generate(self, img_path: str):
        """
        Generate caption for image
        Args:
            img: Image path to generate caption for
        Returns:
            gencap (str): Generated caption
        """

        if self.verbose:
            logger.info(">> Generating caption\n")

        img = Image.open(img_path).convert("RGB")
        image = self.vis_processors["eval"](img).unsqueeze(0).to(self.device)
        gencap = (
            "a photo of "
            + self.model.generate(
                {"image": image},
                min_length=self.min_length,
                max_length=self.max_length,
                repetition_penalty=self.repetition_penalty,
            )[0]
        )
        if self.verbose:
            logger.info(f"=> Generated caption: {gencap}")
        return gencap


if __name__ == "__main__":
    # example usage
    args = argparse.Namespace(
        img_path="data/merlion.png",
    )
    VERBOSE = True
    accelerator = Accelerator()
    device = accelerator.device
    caption_generator = CaptionGenerator(args, device, verbose=VERBOSE)
    if VERBOSE:
        logger = get_logger()
        logger.info(f"=> Generating caption for {args.img_path}")
    img = Image.open(args.img_path)
    generated_caption = caption_generator.generate(img)
