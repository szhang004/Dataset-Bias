import torchvision
import numpy as np


def get_dataset_cifarc(path, ctype):
    count_dict = {}
    count_dict["val"] = {}

    val_images = np.load(path + "/" + ctype + ".npy") 
    val_labels = np.load(path + "/labels.npy") 
    
    dataset = {}
    dataset["val"] = []
    
    for i in range(len(val_images)):
        dataset["val"].append({})
        dataset["val"][i]["image"] = val_images[i]
        dataset["val"][i]["label"] = val_labels[i]

    return dataset, None, count_dict