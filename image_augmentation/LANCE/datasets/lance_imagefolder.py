import os
import csv
import json
import pandas as pd

import torchvision


class LanceImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, data_dir):
        super(LanceImageFolder, self).__init__(data_dir)
        self.data_dir = data_dir
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.csv_field_names = [
            "Image",
            "Label",
            "Reconstruction",
            "LANCE",
            "Caption",
            "Edited Caption",
            "Original",
            "Edit",
            "Edit Type",
            "Modification",
        ]
        self.csv_path = os.path.join(self.data_dir, "data.csv")
        self.generate_csv()
        self.df = pd.read_csv(self.csv_path)

    def generate_csv(self):
        """
        Generates a csv file with the following columns
        tgt: target class
        img_path: path to image relative to data_dir
        lance_path: path to image relative to lance_path
        prompt: original generated caption
        edit: edit to caption
        """
        # Load prompt.txt

        with open(os.path.join(self.data_dir, "data.csv"), "w") as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=self.csv_field_names)
            csvwriter.writeheader()
            for cls in self.idx_to_class.values():
                cls_path = os.path.join(self.data_dir, cls)
                for img_path in os.listdir(cls_path):
                    # Skip if is not a directory
                    if not os.path.isdir(os.path.join(cls_path, img_path)):
                        continue
                    if not os.path.exists(
                        os.path.join(cls_path, img_path, "prompt_dict.json")
                    ):
                        continue
                    with open(
                        os.path.join(cls_path, img_path, "prompt_dict.json")
                    ) as f:
                        prompt_dict = json.load(f)
                        if len(prompt_dict["edits"]) == 0:
                            continue
                    full_img_path = os.path.join(cls_path, img_path)
                    caption = prompt_dict["caption"]
                    source_img_path = full_img_path + "/img.jpeg"
                    recon_img_path = full_img_path + "/img_inv.jpeg"
                    for lance, lance_info in prompt_dict["edits"].items():
                        csvwriter.writerow(
                            {
                                "Image": source_img_path,
                                "Caption": caption,
                                "Label": cls,
                                "Reconstruction": recon_img_path,
                                "LANCE": lance,
                                "Edit": lance_info["edit"],
                                "Edit Type": lance_info["edit_type"],
                                "Original": lance_info["original"],
                                "Edited Caption": lance_info["edited_caption"],
                            }
                        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        img = entry["Image"]
        lance = entry["LANCE"]
        label = entry["Label"]
        return img, lance, label
