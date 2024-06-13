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

import argparse
import json
import os
from PIL import Image
from torch.utils.data import Dataset

class HardImageNet(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.class_map = {
            "n03218198": "Dog Sled",
            "n02492660": "Howler Monkey",
            "n04162706": "Seat Belt",
            "n04228054": "Ski",
            "n04356056": "Sunglasses",
            "n02807133": "Swimming Cap",
            "n02777292": "Balance Beam",
            "n03535780": "Gymnastic Horizontal Bar",
            "n03899768": "Patio",
            "n04019541": "Hockey Puck",
            "n03770439": "Miniskirt",
            "n04264628": "Keyboard Space Bar",
            "n04540053": "Volleyball",
            "n09835506": "Baseball Player",
            "n04251144": "Snorkel",
        }
        self.num_classes = len(self.class_map)
        self.wnid_to_idx = json.load(open('datasets/hard_imagenet_wnid_to_idx.json', 'rb'))
        
        with open("datasets/hard_imagenet_imglst.txt", "r") as f:
            self.img_paths = [img.strip() for img in f.readlines()]

    def map_wnid_to_label(self, wnid):
        raise self.wnid_to_idx[wnid] if wnid in self.wnid_to_idx else -1

    def map_wnid_to_name(self, wnid):        
        return self.class_map[wnid] if wnid in self.class_map else "NA"

    def __getitem__(self, ind):
        wnid_fname = self.img_paths[ind].split(" ")
        wnid = wnid_fname[-2]
        fname = wnid_fname[-1]

        img_path_full = os.path.join(self.data_dir, wnid, fname)
        return img_path_full, self.map_wnid_to_name(wnid)

    def __len__(self):
        return len(self.img_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/srv/share/datasets/ImageNet/val/",
        help="Path to ImageNet dataset",
    )
    args = parser.parse_args()
    dset = HardImageNet(args.img_dir)
    print(dset[0])