import argparse
import json
import os
from PIL import Image
import pdb
import ast
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import pdb
import pandas as pd
class CustomImageFolder(Dataset):
    def __init__(self,data_dir, csv_path, dataset='all'):
        # super(CustomImageFolder, self).__init__(data_dir)
        self.filenames = []
        self.cls = []
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_path)

        if(dataset == 'all'):
            for cls in self.df['class']:
                try:
                    cls = ast.literal_eval(cls)[0]
                except:
                    cls = cls
                self.cls.append(cls)
            self.filenames = self.df['image_path'].tolist()
        else:
            df_sub = self.df[self.df['dataset'] == dataset]
            for cls in df_sub['class']:
                try:
                    cls = ast.literal_eval(cls)[0]
                except:
                    cls = cls
                self.cls.append(cls)
            self.filenames = df_sub['image_path'].tolist()

        # self.filenames = self.filenames[:4]
        # self.csl = self.cls[:4]

        
        # self.filenames = sorted(os.listdir(self.data_dir))
        # self.cls = [data_dir.split('/')[-1]]*len(self.filenames)
        
    def __getitem__(self, ind):
        filename = self.filenames[ind]
        img_path = os.path.join(self.data_dir, filename)
        clsname = self.cls[ind]      
        # print(clsname, ':', img_path) 
        return img_path, clsname

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default="/srv/hoffman-lab/share4/datasets/ADE20K_People/images/orig/person",
        help="Path to custom dataset",
    )
    args = parser.parse_args()
    # pdb.set_trace()
    dset = CustomImageFolder(args.img_dir)
    print(dset[0])