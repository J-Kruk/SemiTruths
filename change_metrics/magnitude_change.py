import cv2
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import math
from skimage import morphology
import matplotlib.colors as mcolors

_ = torch.manual_seed(123)
from scipy.ndimage import label
import pandas as pd
import json
import argparse
import pdb
import os
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from skimage.metrics import structural_similarity
from clip_similarity import *
from dreamsim import dreamsim
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageOps


class Change:
    def __init__(
        self,
        mse_threshold: int = 0.1,
        blip_model_name: str = "Salesforce/blip-image-captioning-large",
        sentence_transormer_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        clip_model_name: str = "ViT-L/14",
        device: str = "cuda",
        save_dir=None,
    ):

        self.mse_threshold = mse_threshold
        # self.perturb_type = perturb_type
        self.device = device

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze")
        self.clip = ClipSimilarity(clip_model_name, device)
        self.dreamsim_model, self.dreamsim_preprocess = dreamsim(pretrained=True)
        self.sen_sim = SentenceTransformer(sentence_transormer_model_name)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            blip_model_name
        )
        self.caption_processor = BlipProcessor.from_pretrained(blip_model_name)
        self.min_component_size = 200
        self.connectivity = 3
        self.min_component_size_1 = 10
        self.distance_threshold = 200
        self.buffer_size = 5
        self.save_dir = save_dir

    def normalize_lpips(self, arr):
        # pdb.set_trace()
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min
        scaled = np.array((arr - arr_min) / max(float(arr_range), 1e-6), dtype="f")
        scaled = np.array((arr - arr_min) / max(float(arr_range), 1e-6), dtype="f")
        arr_new = -1 + (scaled * 2)
        return arr_new

    def get_blip_caption(self, fake):
        # pdb.set_trace()
        text = "a photo of"
        inputs = self.caption_processor(fake, text, return_tensors="pt").to(self.device)
        out = self.caption_model.generate(**inputs)
        caption_og = self.caption_processor.decode(out[0], skip_special_tokens=True)
        return caption_og

    def calc_embeddings(self, fake_img_path="", original_caption="", edited_caption=""):
        # pdb.set_trace()
        # if(self.perturb_type == 'p2p'):
        #     caption_json_path = fake_img_path.replace(os.path.join(fake_img_path.split('/')[-3],fake_img_path.split('/')[-2],fake_img_path.split('/')[-1]), os.path.join('edited_captions',fake_img_path.split('/')[-3],(fake_img_path.split('/')[-2]+'.json')))
        #     with open(caption_json_path, 'r') as file:
        #         caption_json = json.load(file)

        #     for caption in caption_json:
        #         if(caption['edit'] ==fake_img_path.split('/')[-1].split('.')[0]):
        #             original_caption = caption['original_caption']
        #             edited_caption = caption['edited_caption']
        #             break
        sentences = [original_caption, edited_caption]
        return sentences, original_caption

    def calc_sen_sim(self, sentences):
        # pdb.set_trace()type(real)
        embeddings = self.sen_sim.encode(sentences)
        val = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        sem_sim_caption = 1 - max(val.numpy()[0][0], 0)

        return sem_sim_caption

    def calc_ssim(self, real, fake):
        # pdb.set_trace()
        # pdb.set_trace()
        real = np.array(real)
        fake = np.array(fake)
        score, diff = structural_similarity(
            real, fake, channel_axis=-1, win_size=5, full=True
        )
        # diff = (diff * 255).astype("uint8")

        return score

    def calc_mse(self, real, fake):
        # pdb.set_trace()
        real = np.array(real)
        fake = np.array(fake)
        if len(real.shape) == 3:
            mse = np.linalg.norm(real - fake, axis=2)
            mse = (
                np.square(
                    real[:, :, 0].astype(np.float32) - fake[:, :, 0].astype(np.float32)
                )
                + np.square(
                    real[:, :, 1].astype(np.float32) - fake[:, :, 1].astype(np.float32)
                )
                + np.square(
                    real[:, :, 2].astype(np.float32) - fake[:, :, 2].astype(np.float32)
                )
            )
            mse = mse / (255 * 255 * 3)
        if len(real.shape) == 3:
            # mse = np.linalg.norm(real - fake, axis=2)
            mse = (
                np.square(
                    real[:, :, 0].astype(np.float32) - fake[:, :, 0].astype(np.float32)
                )
                + np.square(
                    real[:, :, 1].astype(np.float32) - fake[:, :, 1].astype(np.float32)
                )
                + np.square(
                    real[:, :, 2].astype(np.float32) - fake[:, :, 2].astype(np.float32)
                )
            )
            mse = mse / (255 * 255 * 3)
        else:
            mse = np.square(real.astype(np.float32) - fake.astype(np.float32))
            mse = mse / (255 * 255)
        # mask = (mse*255).astype(np.uint8)
        # binary_image = (mask > self.mse_threshold).astype(np.uint8)
        # print('THRESH', self.mse_threshold)
        mask = mse > self.mse_threshold
        binary_image = (mask * 255).astype(np.uint8)
        # mask = (mse*255).astype(np.uint8)
        # binary_image = (mask > int(255*0.02)).astype(np.uint8)

        return np.mean(mse.flatten()), binary_image, mse

    def calc_dreamsim(self, real, fake):
        # pdb.set_trace()
        real = self.dreamsim_preprocess(real).to(self.device)
        fake = self.dreamsim_preprocess(fake).to(self.device)
        distance = self.dreamsim_model(real, fake)

        return distance.item()

    # def diffused_localized(self, num_labels, labels):
    #     polygons = []
    #     for i in range(1, num_labels):  # Skipping the background label 0
    #         component_mask = (labels == i).astype(np.uint8)
    #         contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         for contour in contours:
    #             if len(contour) >= 3:  # Need at least 3 points to form a polygon
    #                 polygon = Polygon([tuple(point[0]) for point in contour])
    #                 polygons.append(polygon)
    #     buffered_polygons = [poly.buffer(self.buffer) for poly in polygons]  # Adjust buffer distance as needed
    #     intersection_matrix = np.zeros((len(buffered_polygons), len(buffered_polygons)), dtype=int)

    #     for i, poly1 in enumerate(buffered_polygons):
    #         for j, poly2 in enumerate(buffered_polygons):
    #             if i != j and poly1.intersects(poly2):
    #                 intersection_matrix[i, j] = 1
    #     graph = nx.Graph()

    #     for i in range(len(buffered_polygons)):
    #         for j in range(i + 1, len(buffered_polygons)):
    #             if intersection_matrix[i, j] == 1:
    #                 graph.add_edge(i, j)

    #     # Find the number of connected components
    #     num_connected_components = nx.number_connected_components(graph)
    #     if num_connected_components > 1:
    #         return

    def calc_cc_ratio_new(self, binary_image):
        # pdb.set_trace()

        binary_image_1 = np.uint8(
            morphology.remove_small_objects(
                binary_image.astype(bool),
                min_size=self.min_component_size_1,
                connectivity=self.connectivity,
            )
        )
        kernel = np.ones((2 * self.buffer_size + 1, 2 * self.buffer_size + 1), np.uint8)
        dilated_image = cv2.dilate(binary_image_1, kernel)
        # num_labels_1, labels_1, stats_1, centroids_1 = cv2.connectedComponentsWithStats(dilated_image)
        # labels_1 = np.uint8(labels_1)
        # labels_1[labels_1>0] = 1
        # labeled_image_1 = np.zeros_like(labels_1)
        # new_label_1 = 1
        # for i in range(1, num_labels_1):
        #     if stats_1[i, cv2.CC_STAT_AREA] >= self.min_component_size_1:
        #         labeled_image_1[labels_1 == i] = new_label_1
        #         new_label_1 += 1

        num_labels_2, labels_2, stats_2, centroids_2 = cv2.connectedComponentsWithStats(
            dilated_image
        )
        labels_2 = np.uint8(labels_2)

        # Initialize a new label array for the merged components
        labeled_image_2 = np.zeros_like(labels_2)

        # Function to compute Euclidean distance
        def euclidean_distance(point1, point2):
            return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        # Keep track of merged component mappings
        component_mapping = {i: i for i in range(num_labels_2)}

        # Merge components that are close to each other
        for i in range(1, num_labels_2):
            for j in range(i + 1, num_labels_2):
                dist = euclidean_distance(centroids_2[i], centroids_2[j])
                if dist < self.distance_threshold:
                    # Merge component j into component i
                    component_mapping[j] = component_mapping[i]

        # Create the new labeled image with merged components
        for i in range(1, num_labels_2):
            labeled_image_2[labels_2 == i] = component_mapping[i]

        labeled_image_2[labeled_image_2 > 0] = 1
        labeled_image_3 = morphology.remove_small_objects(
            labeled_image_2.astype(bool),
            min_size=self.min_component_size,
            connectivity=self.connectivity,
        ).astype(int)
        # num_labels_3, labels_3, stats_3, centroids_3 = cv2.connectedComponentsWithStats(labeled_image_2)
        # labeled_image_3 = np.zeros_like(labels_3)
        # new_label_3 = 1
        # for i in range(1, num_labels_3):
        #     if stats_3[i, cv2.CC_STAT_AREA] >= self.min_component_size:
        #         labeled_image_3[labels_3 == i] = new_label_3
        #         new_label_3 += 1
        labeled_image_3 = np.uint8(labeled_image_3)
        # labeled_image_3[labeled_image_3>0]=1
        num_final_labels_4, final_labels_4, stats_4, centroids_4 = (
            cv2.connectedComponentsWithStats(labeled_image_3)
        )
        # pdb.set_trace()
        total_pixels = np.sum(stats_4[1:, cv2.CC_STAT_AREA])
        ratio_pixels = total_pixels / (
            labeled_image_3.shape[0] * labeled_image_3.shape[1]
        )

        if num_final_labels_4 > 2:
            distances = pdist(centroids_4[1:], "euclidean")
            max_dist = np.max(distances)
        else:
            max_dist = -1
        if num_final_labels_4 > 1:
            largest_component_size = np.max(stats_4[1:, cv2.CC_STAT_AREA])
        else:
            largest_component_size = 0

        return (
            ratio_pixels,
            final_labels_4,
            num_final_labels_4,
            max_dist,
            total_pixels,
            largest_component_size,
            dilated_image,
        )

    def calc_cc_ratio(self, binary_image):
        # pdb.set_trace()
        if len(np.unique(binary_image)) < 2:
            return 0, 0

        num_labels, labeled_image = cv2.connectedComponents(binary_image)
        flattened_image = labeled_image.flatten()

        label_counts = np.bincount(flattened_image)
        ratio_pixels = 0
        for label, count in enumerate(label_counts[1:]):
            if count < self.min_component_size:
                labeled_image[labeled_image == label + 1] = 0
            else:
                ratio_pixels += count
        largest_component_size = np.max(label_counts[1:])
        return ratio_pixels / (flattened_image.shape[0]), labeled_image

    def plot_cc(self, labeled_image):
        # pdb.set_trace()
        num_unique_values = (
            np.max(labeled_image) + 1
        )  # Number of unique values including 0
        colors = plt.cm.get_cmap(
            "tab20", num_unique_values - 1
        )  # Choose a colormap and get unique colors, excluding 0
        black_color = np.array([[0, 0, 0, 1]])  # Black color for label 0
        colors_array = colors(
            np.arange(num_unique_values - 1)
        )  # Convert colormap object to numpy array
        colors = np.vstack((black_color, colors_array))  # Add black color for label 0
        cmap = mcolors.ListedColormap(colors)
        return cmap

    def save_img(
        self,
        img_real_rgb,
        img_fake_rgb,
        img_fake_gray,
        mse_all_ch,
        mse_gray,
        labeled_image_all_ch,
        labeled_image_gray,
        cc_gray,
        cc_rgb,
        mse_val_rgb,
        mse_val_gray,
        label_counts_rgb,
        label_counts_gray,
        max_dist_rgb,
        max_dist_gray,
        total_pixels_rgb,
        total_pixels_gray,
        dilated_image,
        path,
    ):
        # pdb.set_trace()
        if isinstance(labeled_image_all_ch, int):
            labeled_image_all_ch = np.zeros((mse_all_ch.shape[0], mse_all_ch.shape[1]))
            cmap_all_ch = "gray"
        else:
            cmap_all_ch = self.plot_cc(labeled_image_all_ch)
        if isinstance(labeled_image_gray, int):
            labeled_image_gray = np.zeros((mse_gray.shape[0], mse_gray.shape[1]))
            cmap_gray = "gray"
        else:
            cmap_gray = self.plot_cc(labeled_image_gray)

        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes[0, 0].imshow(cv2.cvtColor(np.array(img_real_rgb), cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("original image ")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(cv2.cvtColor(np.array(img_fake_rgb), cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("changed image")
        axes[0, 1].axis("off")
        axes[0, 2].imshow(np.array(dilated_image), cmap="gray")
        axes[0, 2].set_title("changed image grayscale")
        axes[0, 2].axis("off")

        im = axes[1, 0].imshow(mse_all_ch, cmap="hot", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=axes[1, 0])
        cbar.set_label("MSE  ")
        axes[1, 0].legend(["MSE"], loc="upper right")
        axes[1, 0].imshow(mse_all_ch, cmap="hot")
        txt = "mse visualization(all channels)" + str(mse_val_rgb.item())
        axes[1, 0].set_title(txt)
        axes[1, 0].axis("off")

        axes[1, 1].imshow(labeled_image_all_ch, cmap=cmap_all_ch)
        txt = (
            "connected components(all channels) "
            + str(round(cc_rgb, 3))
            + " : "
            + str(label_counts_rgb)
            + " : "
            + str(max_dist_rgb)
            + " : "
            + str(total_pixels_rgb)
        )
        axes[1, 1].set_title(txt)
        axes[1, 1].axis("off")
        axes[1, 2].hist(mse_all_ch.flatten())
        axes[1, 2].set_xlabel("mse values(all_channels)")
        axes[1, 2].set_ylabel("Frequency")
        axes[1, 2].set_title("Histogram of all values in mse")

        im = axes[2, 0].imshow(mse_gray, cmap="hot", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=axes[2, 0])
        cbar.set_label("MSE  ")
        axes[2, 0].legend(["MSE"], loc="upper right")
        axes[2, 0].imshow(mse_gray, cmap="hot")
        txt = "mse visualization(grayscale)" + str(mse_val_gray.item())
        axes[2, 0].set_title(txt)
        axes[2, 0].axis("off")
        txt = (
            "connected components(gray) "
            + str(round(cc_gray, 3))
            + " : "
            + str(label_counts_gray)
            + " : "
            + str(max_dist_gray)
            + " : "
            + str(total_pixels_gray)
        )
        axes[2, 1].imshow(labeled_image_gray, cmap=cmap_gray)
        axes[2, 1].set_title(txt)
        axes[2, 1].axis("off")
        axes[2, 2].hist(mse_gray.flatten())
        axes[2, 2].set_xlabel("mse values(grayscale)")
        axes[2, 2].set_ylabel("Frequency")
        axes[2, 2].set_title("Histogram of all values in mse")

        plt.tight_layout()
        plt.savefig(path)

    def calc_metrics(
        self, real_img_path, fake_img_path, og_caption="", edit_caption=""
    ):
        # pdb.set_trace()
        img_real_rgb = Image.open(real_img_path).convert("RGB")
        img_fake_rgb = Image.open(fake_img_path).convert("RGB")
        has_nan_real = np.isnan(np.array(img_real_rgb).astype(np.float32)).any()
        has_nan_fake = np.isnan(np.array(img_fake_rgb).astype(np.float32)).any()
        if (has_nan_fake) or (has_nan_real):
            res = ["NA"] * 16
            return res
        else:
            img_fake_rgb = img_fake_rgb.resize(img_real_rgb.size)

            img_real_gray = ImageOps.grayscale(img_real_rgb)
            img_fake_gray = ImageOps.grayscale(img_fake_rgb)

            # Semantic Image Space
            dreamsim = self.calc_dreamsim(img_real_rgb, img_fake_rgb)
            lpips_score = (
                self.lpips(
                    torch.from_numpy(self.normalize_lpips(img_real_rgb))
                    .permute(2, 0, 1)
                    .unsqueeze(0),
                    torch.from_numpy(self.normalize_lpips(img_fake_rgb))
                    .permute(2, 0, 1)
                    .unsqueeze(0),
                )
            ).item()

            # Semantic Caption space
            # if(perturb_type =='p2p'):
            #     sentences, og_caption = self.calc_embeddings(fake_img_path=fake_img_path)
            # else:
            sentences, og_caption = self.calc_embeddings(
                original_caption=og_caption, edited_caption=edit_caption
            )
            sen_sim = self.calc_sen_sim(sentences)
            clip_sim = (self.clip.text_similarity(sentences[0], sentences[1])).item()
            clip_sim = 1 - max(clip_sim, 0)
            # post_caption = self.get_blip_caption(img_fake_rgb)
            # sentences_post = [og_caption, post_caption]
            # sen_sim_post = self.calc_sen_sim(sentences_post)

            # Image size space

            mse_rgb, binary_rgb, mse_img_rgb = self.calc_mse(img_real_rgb, img_fake_rgb)
            mse_gray, binary_gray, mse_img_gray = self.calc_mse(
                img_real_gray, img_fake_gray
            )
            ssim_rgb = self.calc_ssim(img_real_rgb, img_fake_rgb)
            ssim_gray = self.calc_ssim(img_real_gray, img_fake_gray)
            # pdb.set_trace()
            (
                ratio_rgb,
                labeled_image_rgb,
                cc_clusters_rgb,
                cluster_dist_rgb,
                total_pixels_rgb,
                largest_component_size_rgb,
                dilated_image,
            ) = self.calc_cc_ratio_new(binary_rgb)
            (
                ratio_gray,
                labeled_image_gray,
                cc_clusters_gray,
                cluster_dist_gray,
                total_pixels_gray,
                largest_component_size_gray,
                _,
            ) = self.calc_cc_ratio_new(binary_gray)

            if self.save_dir:
                save_path = os.path.join(
                    self.save_dir,
                    "images_new_31",
                    fake_img_path.split("/")[-3],
                    fake_img_path.split("/")[-2],
                )
                if not (os.path.exists(save_path)):
                    os.makedirs(save_path)
                self.save_img(
                    img_real_rgb,
                    img_fake_rgb,
                    img_fake_gray,
                    mse_img_rgb,
                    mse_img_gray,
                    labeled_image_rgb,
                    labeled_image_gray,
                    ratio_gray,
                    ratio_rgb,
                    mse_rgb,
                    mse_gray,
                    cc_clusters_rgb,
                    cc_clusters_gray,
                    cluster_dist_rgb,
                    cluster_dist_gray,
                    total_pixels_rgb,
                    total_pixels_gray,
                    dilated_image,
                    os.path.join(save_path, fake_img_path.split("/")[-1]),
                )

            return [
                dreamsim,
                lpips_score,
                sen_sim,
                clip_sim,
                mse_rgb,
                mse_gray,
                ssim_rgb,
                ssim_gray,
                ratio_rgb,
                ratio_gray,
                largest_component_size_gray,
                largest_component_size_rgb,
                cc_clusters_rgb,
                cc_clusters_gray,
                cluster_dist_rgb,
                cluster_dist_gray,
            ]
