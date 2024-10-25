import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import torchvision
from data.Dataset import ImageDataset, transform, custom_collate_fn


class celebABlond(Dataset):
    def __init__(self, celebA, transform=None):
        self.dataset = celebA
        self.transform = transform
    
    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img = self.dataset[index][0]
        lab = self.dataset[index][1][9]
        if self.transform:
            img = self.transform(img)
        return img, lab


def get_dataset_celebA(path):
    
    label_dict = {}
    count_dict = {}
    count_dict["train"] = {}
    count_dict["val"] = {}
    count_dict["test"] = {}
    train_dataset = torchvision.datasets.CelebA(root = path, download= True)
    print(train_dataset)
    val_dataset = torchvision.datasets.CelebA(root = path, download= False, split="valid")
    test_dataset = torchvision.datasets.CelebA(root = path, download= False, split="test")
    for i in range(40):
        label_dict[i] = []
        count_dict["train"][i] = np.zeros(40)
        count_dict["val"][i] = np.zeros(40)
        count_dict["test"][i] = np.zeros(40)
        
    dataset = {"train": [], "val": [], "test": []}
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate_fn)

    for batch in train_loader:
        images, labels = batch
        for img, lab in zip(images, labels):
            dataset["train"].append({"image": img, "label": lab})
            for i in range(len(lab)):
                count_dict["train"][i][int(lab[i])] += 1
                
    for batch in val_loader:
        images, labels = batch
        for img, lab in zip(images, labels):
            dataset["val"].append({"image": img, "label": lab})
            for i in range(len(lab)):
                count_dict["val"][i][int(lab[i])] += 1
                
    for batch in test_loader:
        images, labels = batch
        for img, lab in zip(images, labels):
            dataset["test"].append({"image": img, "label": lab})
            for i in range(len(lab)):
                count_dict["test"][i][int(lab[i])] += 1
    
    #for img, lab in train_dataset:
    #    dataset["train"].append({"image": img, "label": lab})
    #    for i in range(len(lab)): 
    #        count_dict["train"][i][int(lab[i])] += 1

    #for img, lab in val_dataset:
    #    dataset["val"].append({"image": img, "label": lab})
    #    for i in range(len(lab)):
    #        count_dict["val"][i][int(lab[i])] += 1
    
    #for img, lab in test_dataset:
    #    dataset["test"].append({"image": img, "label": lab})
    #    for i in range(len(lab)):
    #        count_dict["test"][i][int(lab[i])] += 1