import os
import random
import json
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from run_llava import eval_model, load_model, eval_model_wo_loading
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import ipdb
import pandas as pd


def load_data(data_dir, out_path):
    """
    Loads the metadata if there was an intermediate save
    produced from a run being terminated pre-maturely.
    Otherwise, will return an empty dictionary object.
    """

    # Loading dataset metadata:
    print(f"Loading data ...")
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            processed = json.load(f)

        # will remove all processed data from meta
        for k, _ in processed.items():
            if k in list(metadata.keys()):
                del metadata[k]

    else:
        processed = {}
    print(f"Loaded data ...")
    return metadata, processed


def df_to_json(hf_data):
    """
    Transforms the metadata from a DataFrame from
    to a json object (which was what the pipeline was
    written for.)
    """
    data_dict = {}
    org_img_ids = hf_data.image_id.unique()

    for img_id in org_img_ids:
        img_data = hf_data.loc[hf_data.image_id == img_id]
        d_ = {}

        d_["image"] = img_data.iloc[0]["image"]
        d_["dataset"] = img_data.iloc[0]["dataset"]
        d_["entities"] = img_data.iloc[0]["entities"]
        d_["class"] = img_data.iloc[0]["class"]
        d_["objects"] = []

        for _, row in img_data:
            m_ = {}
            m_["id"] = row["mask_id"]
            m_["mask"] = row["mask"]
            m_["name"] = row["mask_name"]
            m_["ratio"] = row["ratio"]
            d_["objects"].append(m_)

        data_dict[img_id] = d_

    return data_dict


def load_data_hf(hf_dataset, data_dir, out_path):
    """
    Loads the metadata from huggingface, checks
    if there was an intermediate save produced from
    a run being terminated pre-maturely. Otherwise,
    will return an empty dictionary object.
    """
    hf_data = load_dataset(hf_dataset)

    # loading flat metadata & joining in mask labels:
    metadata = pd.read_csv(os.path.join(data_dir, "metadata_flat.csv"))
    datasets = metadata.dataset.unique()
    # datasets = [
    #     "CelebAHQ",
    #     "ADE20K",
    #     "SUN_RGBD",
    #     "CityScapes",
    #     "HumanParsing",
    #     "OpenImages",
    # ]

    ds_dfs = []
    for ds in datasets:
        data = hf_data[ds]
        data_df = pd.DataFrame(data)
        data_df["dataset"] = [ds] * len(data_df)
        ds_dfs.append(data_df)
    full_data = pd.concat(ds_dfs)
    full_data = full_data.loc[full_name.mask_name != "NA"]
    # full_data = full_data.rename(
    #     columns={"label": "mask", "img_name": "image_id", "mask_name": "mask_id"}
    # )

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            processed = json.load(f)

        # removing already processed data from consideration
        full_data = full_data.loc[~full_data.img_name.isin(processed.img_name)]

    # transforming HF data into dictionary:
    full_data_dict = df_to_json(hf_data)
    return full_data_dict, processed


def sample_data(metadata, n=20):
    """
    Samples n images randomly from the Half-Truths Input Data.
    Returns the sample as a dictionary.
    """
    img_ids = list(metadata.keys())
    exp_id = random.choices(img_ids, k=n)
    data_samp = {}

    for id_ in exp_id:
        info = metadata[id_]
        info["object"] = random.choice(info["objects"])
        del info["objects"]
        data_samp[id_] = info

    return data_samp


def sample_by_dataset(metadata, ds):
    """
    Samples images from the Half-Truths Input Data that
    belong to the specific benchmark `ds`. Returns a dictionary.
    """
    print(f"Sampling input data from {ds}...")
    data_samp = {}
    for k, v in tqdm(metadata.items(), total=len(metadata)):
        if v["dataset"] == ds:
            data_samp[k] = v

    assert len(data_samp), f"{ds} is not one of the datasets within Half-Truths"
    return data_samp


def clean_json_output(output):
    """
    Cleans the model's JSON object output into something that
    will be read by json.loads.

    Return json object.
    """

    if "`" in output or "json" in output:
        output = output[output.find("{") : output.rfind("}") + 1]
        output = output.replace("`", "")
    output = output.replace("\n", "").replace("\\", "")

    return json.loads(output)


def visualize_mask_label_perts(
    data_samp, root_dir, viz_dir, figsize=(15, 15), suptitle=None
):
    """
    This function visualizes a set of 6 images using matplotlib.
    """

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    ids = list(data_samp.keys())
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        image_info = data_samp[ids[i]]

        try:
            target_str = image_info["object"]["target"]
            mag_str = (
                f'Magnitude: {image_info["object"]["sem_magnitude"]} semantic change'
            )
            mask_name = image_info["object"]["name"]
        except KeyError:
            obj = random.choice(image_info["objects"])
            target_str = obj["target"]
            mag_str = f'Magnitude: {obj["sem_magnitude"]} semantic change'
            mask_name = obj["name"]

        title_ = f"Recommended Change(s):\n{mag_str}\n{target_str}"
        image_ = Image.open(os.path.join(root_dir, image_info["image_path"]))

        ax.imshow(image_)
        ax.set_title(title_, fontsize=10)
        ax.set_xlabel(f"Mask: {mask_name}", fontweight="bold")
        ax.set_ylabel(f"Dataset: {image_info['dataset']}", fontweight="bold")
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])
        ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "llava_hermes_perturb_viz.png"))
    plt.show()


def log_pert_quality(log_data, outfile="pert_quality_log.txt"):
    """
    Prints the set of data contained in log_data into
    a log file to enable quality inspection.
    """

    with open(outfile, "w") as log_file:
        log_file.write(f"TIMESTAMP:  {datetime.now()}\n\n")
        log_objs = []
        for _, values in log_data.items():
            try:
                obj = random.choice(values["objects"])
                log_file.write(f"Mask label:   {obj['name']}\n")
                log_file.write(f"Sem Change Mag:   {obj['sem_magnitude']}\n")
                log_file.write(f"Mask Pert:   {obj['target']}\n\n")
                log_objs.append(obj)
            except:
                continue

        qual_flags_total = len([v for v in log_objs if v["target_qual_flag"]])
        log_file.write(f"Number of quality flags in sample:   {qual_flags_total}\n")


def verify_output(output, mask_label, counter):
    """
    Verifies that LLaVA perturbed caption is the correct
    length and form. Returns counter and boolean confirming
    whether the output is quality.
    """
    words = output.split()
    if (
        output == ""
        or len(words) > 8
        or len(output) > 200
        or "add " in output[:5].lower()
        or "adding " in output[:8].lower()
        or ":" in output
        or mask_label.lower().strip() == output.lower().strip()
    ):
        return counter + 1, False
    else:
        return counter + 1, True


def mask_change_inference_v0(
    img_path,
    mask,
    mag_change,
    context,
    args,
    tokenizer,
    model,
    image_processor,
    context_len,
):
    """
    Combines all parts of the LLaVA prompts and
    queries the model for inference. Returns the model's
    output, which is the recommended mask augmentation.
    """

    mask = mask.replace("_", " ")
    prompt_types = ["aug", "replace"]
    type_ = random.choice(prompt_types)
    ### augmentation prompt
    if type_ == "aug":
        prompt = (
            f"{context} what augmentation can I make to {mask} in the image that would produce a {mag_change} semantic change? The change must be of a similar size and structure to the {mask}. "
            + f"Represent the recommended change with a preposition, as in '{mask} with X', or with an adjective, as in 'X {mask}'. "
            + f"Do not include explanation and use as few words as possible."
        )

    ### change prompt
    else:
        prompt = (
            f"{context} what is something of a similar size and structure to the {mask} that I can replace the {mask} with to produce a {mag_change} semantic change in the image? "
            + f"Return only the target object. Do not include explanation and  as few words as possible."
        )

    setattr(args, "query", prompt)
    setattr(args, "image_file", img_path)

    return eval_model_wo_loading(tokenizer, model, image_processor, context_len, args)


def mask_change_inference_v1(
    img_path,
    mask,
    mag_change,
    context,
    args,
    tokenizer,
    model,
    image_processor,
    context_len,
):
    """
    Combines all parts of the LLaVA prompts and
    queries the model for inference. Returns the model's
    output, which is the recommended mask augmentation.
    """

    mask = mask.replace("_", " ")
    prompt_types = ["aug", "replace"]
    type_ = random.choice(prompt_types)
    ### augmentation prompt
    if type_ == "aug":
        prompt = (
            f"{context} how can the {mask} in the image be altered to would produce a {mag_change} semantic change? The change must be of a similar size and structure to the {mask}. "
            + f"Represent the recommended change with a preposition, as in '{mask} with X', or with an adjective, as in 'ADJ {mask}'. "
            + f"Concretely state how the {mask} should be augmented, avoid answers such as '{mask} from a different perspective'."
            + f"Do not include explanation and use as few words as possible."
        )

    ### change prompt
    else:
        prompt = (
            f"{context} what is something of a similar size and structure to the {mask} that I can replace the {mask} with to produce a {mag_change} semantic change in the image? "
            + f"Return only the target object. Do not include explanation and  as few words as possible."
        )

    setattr(args, "query", prompt)
    setattr(args, "image_file", img_path)

    return eval_model_wo_loading(tokenizer, model, image_processor, context_len, args)


def mask_change_inference(
    img_path,
    mask,
    mag_change,
    context,
    args,
    tokenizer,
    model,
    image_processor,
    context_len,
):
    """
    Combines all parts of the LLaVA prompts and
    queries the model for inference. Returns the model's
    output, which is the recommended mask augmentation.
    """

    mask = mask.replace("_", " ")
    prompt_types = ["aug", "replace"]
    type_ = random.choice(prompt_types)
    ### augmentation prompt
    if type_ == "aug":
        prompt = (
            f"{context} how can the {mask} in the image be altered to would produce a {mag_change} semantic change? The change must be of a similar size and structure to the {mask}. "
            + f"Represent the recommended change with a preposition, as in '{mask} with X', or with an adjective, as in 'X {mask}'. "
            + f"Concretely identify the color, texture, or attribute augmentation in your recommendation."
            + f"Do not include explanation and use as few words as possible."
        )

    ### change prompt
    else:
        prompt = (
            f"{context} what is something of a similar size and structure to the {mask} that I can replace the {mask} with to produce a {mag_change} semantic change in the image? "
            + f"Return only the target object. Do not include explanation and use as few words as possible."
        )

    setattr(args, "query", prompt)
    setattr(args, "image_file", img_path)

    return eval_model_wo_loading(tokenizer, model, image_processor, context_len, args)


def get_llava_perts(
    data,
    processed_metadata,
    root_dir,
    outpath,
    context,
    args,
    tokenizer,
    model,
    image_processor,
    context_len,
    viz_dir,
    retry=3,
):
    """
    Uses the latest instantiation of the ``size_based_change`` function
    to generate recommended perturbations for a set of images using LLaVA.

    The recommended perturbation is saved in the ``data_samp`` dictionary,
    under the key ``target``.
    """
    progress = 0
    recent_perts = []
    for id_, info in tqdm(data.items(), total=len(data)):

        if "object" in info.keys():
            mask = info["object"]["name"]
            mask = "".join([i for i in mask if not i.isdigit()])
            try:
                image = os.path.join(root_dir, info["image_path"])
            except:
                image = info["object"]["image"]

            output = ""
            counter = 0
            qual = False
            qual_flag = False
            while qual is False:
                mag_change = random.choice(["small", "medium", "large"])
                output = mask_change_inference(
                    image,
                    mask,
                    mag_change,
                    context,
                    args,
                    tokenizer,
                    model,
                    image_processor,
                    context_len,
                )
                if ":" in output:
                    output = output[output.find(":") + 1 :]

                counter, qual = verify_output(output, mask, counter)
                if counter >= retry:
                    qual = True
                    qual_flag = True

            info["object"]["target"] = output
            info["object"]["sem_magnitude"] = mag_change
            info["target_qual_flag"] = qual_flag
            processed_metadata[id_] = info

        elif "objects" in info.keys():
            new_objects = []
            for obj_info in info["objects"]:
                mask = obj_info["name"]

                try:
                    image = os.path.join(root_dir, info["image_path"])
                except:
                    image = obj_info["image"]

                output = ""
                counter = 0
                qual = False
                qual_flag = False
                while qual is False:
                    mag_change = random.choice(["small", "medium", "large"])
                    output = mask_change_inference(
                        image,
                        mask,
                        mag_change,
                        context,
                        args,
                        tokenizer,
                        model,
                        image_processor,
                        context_len,
                    )
                    if ":" in output:
                        output = output[output.find(":") + 1 :]

                    counter, qual = verify_output(output, mask, counter)
                    if counter >= retry:
                        qual = True
                        qual_flag = True

                obj_info["target"] = output
                obj_info["sem_magnitude"] = mag_change
                obj_info["target_qual_flag"] = qual_flag
                new_objects.append(obj_info)

            info["objects"] = new_objects

            # sanity check, making sure all masks are processed:
            if len(new_objects) != len(info["objects"]):
                print("We lost an object somehwere!!")
                ipdb.set_trace()
            processed_metadata[id_] = info

        recent_perts.append(id_)

        if progress % 10 == 0 and progress > 0:
            # saving interm mask label pert file:
            with open(outpath, "w") as out_file:
                json.dump(processed_metadata, out_file)

            # print 10 perts for quality check
            log_data = {}
            for k in recent_perts:
                log_data[k] = processed_metadata[k]
            log_pert_quality(log_data, outfile="pert_quality_log.txt")

            # Visualize a sample of 6 perts for quality check:
            viz_data = {}
            samp_ids = random.sample(recent_perts, 6)
            for k in samp_ids:
                viz_data[k] = processed_metadata[k]
            try:
                visualize_mask_label_perts(
                    viz_data, root_dir, viz_dir, figsize=(15, 15), suptitle=None
                )
            except:
                print("visualization failed ...")

            # recent tracking of recent perturbations:
            recent_perts = []

        progress += 1

    # final save:
    with open(outpath, "w") as out_file:
        json.dump(processed_metadata, out_file)

    return processed_metadata


class InpaintingPertDataset(Dataset):
    """
    This is a custiom pytorch dataser used for generating
    augmented images via diffusion inpainting.
    """

    def __init__(self, data_dict):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data_dict
        self.img_ids = list(data_dict.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = idx
        img_info = self.img_ids[idx]
        return img_id, img_info