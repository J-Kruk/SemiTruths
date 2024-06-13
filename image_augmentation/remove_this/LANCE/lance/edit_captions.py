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

import sys
from pathlib import Path

# add to 'lit-llama' folder to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lit-llama"))

import time
import random
import difflib
import lightning as L
import torch
from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup, quantization
from argparse import Namespace
from accelerate.logging import get_logger

logger = get_logger("lance")
from lance.utils.misc_utils import *

torch.set_float32_matmul_precision("medium")

perturb_prompts = {
    "subject": "Generate all possible variations by changing only the subject of the provided sentence.",
    "weather": "Generate all possible variations of the provided sentence by only changing the weather conditions, or adding a description of the weather if not already present.",
    "object": "Generate all possible variations by changing only the object of the provided sentence.",
    "adjective": "Generate all possible variations of the provided sentence by only adding or altering a single adjective or attribute.",
    "background": "Generate all possible variations of the provided sentence by changing or adding background or location details without altering the foreground or its attributes.",
    "light": "Generate all possible variations by changing only the lighting conditions of the provided sentence."
}

class CaptionEditor:
    def __init__(
        self,
        args,
        device,
        verbose=False,
        llama_max_seq_length=1024,
        llama_max_new_tokens=400,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        perturbation_type="all",
    ):
        self.args = args
        self.device = device
        self.verbose = verbose
        self.llama_max_seq_length = llama_max_seq_length
        self.llama_max_new_tokens = llama_max_new_tokens
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.perturbation_type = perturbation_type

    def load_model(self):
        """
        Loads the LLaMA model from the provided checkpoint paths.

        Returns:
            model: Lit-LLaMA model
        """
        lora_path = Path(self.args.llama_finetuned_path)
        pretrained_path = Path(self.args.llama_pretrained_path)
        lora_r = self.lora_r
        lora_alpha = self.lora_alpha
        lora_dropout = self.lora_dropout

        assert lora_path.is_file()
        assert pretrained_path.is_file()
        precision = (
            "bf16-true"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "32-true"
        )
        fabric = L.Fabric(devices=1, precision=precision)
        # fabric.launch()
        if self.verbose:
            logger.info("[Loading fine-tuned LLaMA model]")

        t0 = time.time()

        with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(
            lora_path
        ) as lora_checkpoint:
            name = llama_model_lookup(pretrained_checkpoint)

            with fabric.init_module(empty_init=True), lora(
                r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True
            ):
                model = LLaMA.from_name(name)

                # 1. Load the pretrained weights
                model.load_state_dict(pretrained_checkpoint, strict=False)
                # 2. Load the fine-tuned lora weights
                model.load_state_dict(lora_checkpoint, strict=False)

        if self.verbose:
            logger.debug(f"Time to load model: {time.time() - t0:.02f} seconds.")

        model.eval()
        # model = fabric.setup_module(model)
        model = fabric.setup(model)

        return model

    def load_tokenizer(self):
        """
        Loads the LLaMA tokenizer from the provided path.

        Returns:
            tokenizer: LLaMA tokenizer
        """
        tokenizer_path = Path(self.args.llama_tokenizer_path)
        assert tokenizer_path.is_file()

        tokenizer = Tokenizer(tokenizer_path)
        return tokenizer

    def generate_response(self, instruction, input):
        """
        Generates a response to the provided instruction and input using the provided model.

        Args:
            model: LLaMA model
            instruction: instruction string
            input: input string

        Returns:
            output: response string
        """
        max_seq_length = self.llama_max_seq_length
        sample = {"instruction": instruction, "input": input}
        prompt = self.generate_prompt(sample)
        encoded = self.tokenizer.encode(
            prompt, bos=True, eos=False, device=self.model.device
        )
        output = generate(
            self.model,
            idx=encoded,
            max_seq_length=max_seq_length,
            max_new_tokens=self.llama_max_new_tokens,
            eos_id=self.tokenizer.eos_id,
            # quantize='llm.int8'
        )
        output = self.tokenizer.decode(output)
        return output

    def generate_prompt(self, example):
        """
        Generates a standardized message to prompt the model with an instruction, optional input and a
        'response' field.

        Args:
            example: dictionary with 'instruction' and 'input' fields

        Returns:
            prompt: string
        """
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Variations:"
        )

    def process_responses(self, response, perturbation_type):
        """
        Processes the response from the model to extract the perturbations.

        Args:
            response: string

        Returns:
            perturbations_list_cleaned: list of perturbations
        """
        perturbations_list_cleaned = []
        try:
            perturbations_str = response.split("### Variations:")[1]
            # split by one or more newlines
            perturbations_list = perturbations_str.split("\n")
            perturbations_list = [p.strip() for p in perturbations_list]
            perturbations_list = [p for p in perturbations_list if len(p) > 0]
            # remove ordinal numbers at start of string
            for p in perturbations_list:
                if p[0].isdigit():
                    perturbation = p.split(" ", 1)
                    if len(perturbation) > 1:
                        perturbations_list_cleaned.append(perturbation[1])
                elif len(p) > 1 and p[1] == "." or len(p) > 2 and p[2] == ".":
                    perturbations_list_cleaned.append(p.split(".")[1])
                elif len(p) > 5:
                    perturbations_list_cleaned.append(p)
        except:
            logger.warning("LLM output could not be parsed")
        peturbation_types = [perturbation_type] * len(perturbations_list_cleaned)
        return perturbations_list_cleaned, peturbation_types

    def _compute_diff(self, cap1, cap2):
        """
        Computes the difference between two captions.
        """
        words_changed = list(
            difflib.ndiff(
                cap1.lower().replace(".", "").strip().split(" "),
                cap2.lower().replace(".", "").strip().split(" "),
            )
        )
        diff1, diff1_len, maxdiff1, maxdiff1_len = [], 0, [], 0
        diff2, diff2_len, maxdiff2, maxdiff2_len = [], 0, [], 0
        for word in words_changed:
            if "+" in word and len(word) > 2:
                diff1.append(word[2:])
                diff1_len += 1
                if diff1_len > maxdiff1_len:
                    maxdiff1_len = diff1_len
                    maxdiff1 = diff1
            else:
                diff1_len = 0
                diff1 = []

            if "-" in word and len(word) > 2:
                diff2.append(word[2:])
                diff2_len += 1
                if diff2_len > maxdiff2_len:
                    maxdiff2_len = diff2_len
                    maxdiff2 = diff2
            else:
                diff2_len = 0
                diff2 = []

        if len(maxdiff1):
            edit_word = " ".join(maxdiff1)
        else:
            edit_word = ""

        if len(maxdiff2):
            original_word = " ".join(maxdiff2)
        else:
            original_word = ""

        return original_word, edit_word

    def _parse_perturbations(self, original_cap, perturbed_caps, edit_types):
        """
        Parses the perturbations into a list of dictionaries.
        """
        perturbations_dict_list = []

        for ix, pcap in enumerate(perturbed_caps):
            original, edit = self._compute_diff(original_cap, pcap)
            perturbations_dict_list.append(
                {
                    "original_caption": original_cap,
                    "original": original,
                    "edited_caption": pcap,
                    "edit": edit,
                    "perturbation_type": edit_types[ix],
                }
            )
        return perturbations_dict_list

    def edit(self, caption, perturbation_type="all"):
        """
        Edits the provided caption using the provided perturbation type.

        Args:
            caption: The caption to edit (string)
            perturbation_type: The type of perturbation to apply (string)

        Returns:
            perturbations: list of perturbations
        """
        assert perturbation_type in list(perturb_prompts.keys()) + [
            "all"
        ], "invalid perturbation type"
        if self.verbose:
            logger.info(">> Editing caption\n")

        perturbations, edit_types = [], []
        if perturbation_type == "all":
            perturbation_types = list(perturb_prompts.keys())
        else:
            perturbation_types = [perturbation_type]
        for perturbation_type in perturbation_types:
            response = self.generate_response(
                perturb_prompts[perturbation_type], caption
            )
            cur_perturbations, cur_types = self.process_responses(
                response, perturbation_type
            )
            perturbations += cur_perturbations
            edit_types += cur_types

        perturbed_captions_dict_list = self._parse_perturbations(
            caption, perturbations, edit_types
        )
        random.shuffle(perturbed_captions_dict_list)
        if self.verbose:
            for ix in range(min(len(perturbed_captions_dict_list), 10)):
                ecap = perturbed_captions_dict_list[ix]["edited_caption"]
                logger.info(f"{ix+1}: {ecap}\n")
            logger.info(f"...\n")
        return perturbed_captions_dict_list


if __name__ == "__main__":
    from accelerate import Accelerator
    from accelerate.logging import get_logger

    # example usage
    args = Namespace(
        llama_finetuned_path="checkpoints/caption_editing/lit-llama-lora-finetuned.pth",
        llama_pretrained_path="checkpoints/caption_editing/lit-llama.pth",
        llama_tokenizer_path="checkpoints/caption_editing/tokenizer.model",
    )
    VERBOSE = True
    caption = "A man wearing sunglasses and standing by the river"
    accelerator = Accelerator()
    device = accelerator.device
    if VERBOSE:
        logger = get_logger()
        logger.info(f"=> Caption: {caption} ")
    editor = CaptionEditor(args, device=torch.device("cuda"), verbose=VERBOSE)
    perturbed_captions = editor.edit(caption)
