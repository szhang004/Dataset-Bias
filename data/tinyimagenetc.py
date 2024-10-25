import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torchvision
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_dataset_tinyimagenet_c(path, ctype, cdeg):
    count_dict = {}
    count_dict["val"] = {}

    val_images = []
    val_labels = []
    
    full_path = os.path.join(path, ctype, cdeg)
    label_mapper = create_label_mapper(full_path)
    
    for cl in os.listdir(full_path):
        for file in os.listdir(os.path.join(full_path, cl)):
            val_images.append(mpimg.imread(os.path.join(full_path, cl,file)))
            lab = label_mapper[cl]
            val_labels.append(lab)
            if lab in count_dict.keys(): count_dict[lab] += 1
            else: count_dict[lab] = 1
    dataset = {}
    dataset["val"] = []
    for i in range(len(val_images)):
        dataset["val"].append({})
        dataset["val"][i]["image"] = val_images[i]
        dataset["val"][i]["label"] = val_labels[i]

    return dataset, None, count_dict


def create_label_mapper(path):
    class_folders = sorted(os.listdir(path))  # Sort to ensure consistent label assignment
    label_mapper = {class_folder: idx for idx, class_folder in enumerate(class_folders)}
    return label_mapper
