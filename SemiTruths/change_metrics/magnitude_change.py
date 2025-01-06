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
from dreamsim import dreamsim
from PIL import Image, ImageOps 





class Change:
    '''
    Class to calculate the change metrics between two images. This function calculates the following metrics:
    1. Semantic similarity between the two images using Dreamsim
    2. Semantic similarity between the two images using LPIPS
    3. Semantic similarity between the two captions using Sentence Transformer
    4. Structural similarity between the two images
    5. Mean Squared Error between the two images
    6. Connected Components ratio between the two images
    7. Largest connected component size
    8. Number of connected components
    9. Maximum distance between connected components
    10. Total number of pixels in connected components
    11. Saves the image indicating the change between the two images

    Inputs:
    mse_threshold: float, default = 0.1
        Threshold for Mean Squared Error between the two images
    sentence_transormer_model_name: str, default = "sentence-transformers/all-MiniLM-L6-v2"
        Model name for Sentence Transformer model
    device: str, default = "cuda"
        Device to run the model on
    save_dir: str, default = None
        Directory to save the images indicating the change between the two images
    save: bool, default = False
        Whether to save the image indicating the change between the two images
    '''
    def __init__(
        self,
        mse_threshold: int = 0.1,
        sentence_transormer_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        save_dir=None,
        save = False,
    ):

        self.mse_threshold = mse_threshold
        self.device = device
        self.save = save 

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
        self.dreamsim_model, self.dreamsim_preprocess = dreamsim(pretrained=True)
        self.sen_sim = SentenceTransformer(sentence_transormer_model_name)
        self.min_component_size =  200 # Minimum size of connected component to be considered
        self.connectivity = 3 # Connectivity for connected components
        self.min_component_size_1 = 10 # Minimum size of connected component to be considered
        self.distance_threshold = 200 # Distance threshold for merging connected components
        self.buffer_size = 5  # Buffer size for dilating the binary image
        self.save_dir = save_dir 
    


    def normalize_lpips(self, arr):
        '''
        Function to normalize the image between -1 and 1
        Inputs: arr: np.array, Image to be normalized
        Outputs: arr_new: np.array, Normalized image
        '''
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min
        scaled = np.array((arr-arr_min) / max(float(arr_range),1e-6), dtype='f')
        arr_new = -1 + (scaled * 2)
        return arr_new

    def calc_embeddings(self,fake_img_path='',original_caption='', edited_caption='' ):
        '''
        Function to calculate the embeddings for the images and captions
        Inputs: fake_img_path: str, Path to the fake image
                original_caption: str, Original caption
                edited_caption: str, Edited caption
        Outputs: embeddings: np.array, Embeddings for the images and captions
        '''

        sentences = [original_caption, edited_caption]  
        return sentences, original_caption


    def calc_sen_sim(self,sentences):
        '''
        Function to calculate the semantic similarity between the two captions
        Inputs: sentences: list, List of sentences
        Outputs: sem_sim_caption: float, Semantic similarity between the two captions
        '''
        embeddings = self.sen_sim.encode(sentences)
        val = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        sem_sim_caption = 1-max(val.numpy()[0][0],0)

        return sem_sim_caption

    def calc_ssim(self,real,fake):
        '''
        Function to calculate the structural similarity between the two images
        Inputs: real: np.array, Real image
                fake: np.array, Fake image
        Outputs: score: float, Structural similarity between the two images
        '''
        real = np.array(real)
        fake = np.array(fake)
        score,diff = structural_similarity(real, fake, channel_axis=-1,win_size=5, full=True)

        return score

    def calc_mse(self,real,fake):
        '''
        Function to calculate the Mean Squared Error between the two images
        Inputs: real: np.array, Real image
                fake: np.array, Fake image
        Outputs: mse: float, Mean Squared Error between the two images
                 binary_image: np.array, Binary image indicating the change between the two images
                 mse: np.array, Mean Squared Error between the two images
        '''
        real = np.array(real)
        fake = np.array(fake)
        if(len(real.shape) == 3):
            # mse = np.linalg.norm(real - fake, axis=2)
            mse = np.square(real[:,:,0].astype(np.float32) - fake[:,:,0].astype(np.float32)) + np.square(real[:,:,1].astype(np.float32) - fake[:,:,1].astype(np.float32))+np.square(real[:,:,2].astype(np.float32) - fake[:,:,2].astype(np.float32))
            mse = mse / (255*255*3)
        else:
            mse = np.square(real.astype(np.float32) - fake.astype(np.float32))
            mse = mse/ (255*255)
        mask = (mse > self.mse_threshold)
        binary_image = (mask*255).astype(np.uint8)

        return np.mean(mse.flatten()), binary_image, mse
    
    def calc_dreamsim(self, real, fake):
        '''
        Function to calculate the semantic similarity between the two images using Dreamsim
        Inputs: real: np.array, Real image
                fake: np.array, Fake image
        Outputs: distance: float, Semantic similarity between the two images
        '''

        real = self.dreamsim_preprocess(real).to(self.device)
        fake = self.dreamsim_preprocess(fake).to(self.device)
        distance = self.dreamsim_model(real, fake)

        return distance.item()


    def calc_cc_ratio(self, binary_image):
        '''
        Function to calculate the connected components ratio between the two images
        Inputs: binary_image: np.array, Binary image indicating the change between the two images
        Outputs: ratio_pixels: float, Connected components ratio between the two images
                    labeled_image: np.array, Labeled image indicating the connected components
                    num_labels: int, Number of connected components
                    max_dist: float, Maximum distance between connected components
                    total_pixels: int, Total number of pixels in connected components
                    largest_component_size: int, Largest connected component size
                    dilated_image: np.array, Dilated binary image
                    save_img: np.array, Image indicating the change between the two images
        '''
      
        binary_image_1 = np.uint8(morphology.remove_small_objects(binary_image.astype(bool), min_size=self.min_component_size_1, connectivity=self.connectivity)) # Remove small objects
        kernel = np.ones((2 * self.buffer_size + 1, 2 * self.buffer_size + 1), np.uint8) # Create a kernel for dilation
        dilated_image = cv2.dilate(binary_image_1, kernel) # Dilate the binary image

        num_labels_2, labels_2, stats_2, centroids_2 = cv2.connectedComponentsWithStats(dilated_image) # Get connected components
        labels_2 = np.uint8(labels_2) # Convert labels to uint8
        
        # Initialize a new label array for the merged components
        labeled_image_2 = np.zeros_like(labels_2)

        # Function to compute Euclidean distance
        def euclidean_distance(point1, point2):
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
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

        labeled_image_2[labeled_image_2>0]=1 # Convert to binary image
        labeled_image_3 = morphology.remove_small_objects(labeled_image_2.astype(bool), min_size=self.min_component_size, connectivity=self.connectivity).astype(int) # Remove small objects
        labeled_image_3 = np.uint8(labeled_image_3) # Convert to uint8
        num_final_labels_4, final_labels_4, stats_4, centroids_4 = cv2.connectedComponentsWithStats(labeled_image_3) # Get connected components
        total_pixels = np.sum(stats_4[1:,cv2.CC_STAT_AREA]) # Total number of pixels in connected components
        ratio_pixels = total_pixels/(labeled_image_3.shape[0]*labeled_image_3.shape[1]) # Connected components ratio

        '''
        Calculate the maximum distance between connected components
        '''
        if(num_final_labels_4 > 2):
            distances = pdist(centroids_4[1:], 'euclidean')
            max_dist = np.max(distances)
        else:
            max_dist = -1
        if(num_final_labels_4>1):
            largest_component_size = np.max(stats_4[1:,cv2.CC_STAT_AREA])
        else:
            largest_component_size = 0

        '''
        Get the image indicating the change between the two images
        '''
        save_img = final_labels_4.copy()
        save_img[save_img>0] = 255
        save_img = np.uint8(save_img)
        
        
        return ratio_pixels, final_labels_4, num_final_labels_4, max_dist, total_pixels, largest_component_size, dilated_image, save_img
    
    def calc_metrics(self,real_img_path,fake_img_path,og_caption ='',edit_caption =''):
        '''
        Function to calculate the change metrics between the two images
        Inputs: real_img_path: str, Path to the real image
                fake_img_path: str, Path to the fake image
                og_caption: str, Original caption
                edit_caption: str, Edited caption
        Outputs: res: list, List of change metrics
        '''

        # Load the images, if the images are not present return NA
        try:
            img_real_rgb = Image.open(real_img_path).convert('RGB')
        except:
            res = ['NA']*9
            return res 
        try:
            img_fake_rgb = Image.open(fake_img_path).convert('RGB')
        except:
            res = ['NA']*9
            return res 
        # Check if the images have NaN values
        has_nan_real = np.isnan(np.array(img_real_rgb).astype(np.float32)).any() 
        has_nan_fake = np.isnan(np.array(img_fake_rgb).astype(np.float32)).any()
        if(has_nan_fake) or (has_nan_real):
            res = ['NA']*16
            return res 
        else:
            img_fake_rgb =img_fake_rgb.resize(img_real_rgb.size)

            img_real_gray=ImageOps.grayscale(img_real_rgb)

            #Semantic Image Space
            dreamsim = self.calc_dreamsim(img_real_rgb, img_fake_rgb)
            lpips_score = (self.lpips(torch.from_numpy(self.normalize_lpips(img_real_rgb)).permute(2, 0, 1).unsqueeze(0), torch.from_numpy(self.normalize_lpips(img_fake_rgb)).permute(2, 0, 1).unsqueeze(0))).item()

            #Semantic Caption space
            sentences, og_caption = self.calc_embeddings(original_caption=og_caption, edited_caption=edit_caption)
            sen_sim = self.calc_sen_sim(sentences)
            
            #Image size space
            mse_rgb, binary_rgb, mse_img_rgb = self.calc_mse(img_real_rgb, img_fake_rgb)
            ssim_rgb = self.calc_ssim(img_real_rgb,img_fake_rgb)
            ratio_rgb, labeled_image_rgb, cc_clusters_rgb, cluster_dist_rgb, total_pixels_rgb, largest_component_size_rgb, dilated_image, save_img = self.calc_cc_ratio(binary_rgb)
            if(self.save):
                name = os.path.join(self.save_dir, fake_img_path.split('/')[-3], fake_img_path.split('/')[-2])
                if(not os.path.exists(name)):
                    os.makedirs(name)
                save_img = Image.fromarray(save_img)
                save_img.save(os.path.join(name, os.path.splitext(fake_img_path.split('/')[-1])[0]))

            return [ratio_rgb, ssim_rgb, mse_rgb,lpips_score, dreamsim, sen_sim, largest_component_size_rgb, cc_clusters_rgb, cluster_dist_rgb]








