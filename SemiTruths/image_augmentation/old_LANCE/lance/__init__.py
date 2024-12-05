import sys
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from .generate_captions import *
from .edit_captions import *
from .edit_images import *
from .utils.misc_utils import *
from .utils.inference_utils import predict

sys.path.append("../")
sys.path.append("../datasets")

try:
    import datasets.lance_imagefolder as lif
    import datasets.custom_imagefolder as cif
except ModuleNotFoundError:
    import SemiTruths.image_augmentation.LANCE.datasets.lance_imagefolder as lif
    import SemiTruths.image_augmentation.LANCE.datasets.custom_imagefolder as cif
