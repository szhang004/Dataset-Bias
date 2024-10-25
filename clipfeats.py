from datasets import load_dataset
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
from torch.utils.data import DataLoader
from data.celebA import get_dataset_celebA
from data.cifar100 import get_dataset_cifar
from data.imagenet import get_dataset_imagenet
from train import get_dataset
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
import csv
from data.Dataset import ImageDataset, transform, custom_collate_fn
from torchvision.transforms import Lambda




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--cset", type=str, default="train")
    args = parser.parse_args()

    dataset, bbox_data, counts = get_dataset(args)
    ds = ImageDataset(dataset, args.cset, transform)
    loader = DataLoader(dataset[args.cset], batch_size=4, collate_fn = custom_collate_fn, num_workers = 8)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_feats = []
    # Iterate over data.
    with open("CLIP_"+args.dataset+"_"+args.cset, 'w+') as csvfile:
        writer = csv.writer(csvfile)
        i = 0
        for (inputs, labels) in iter(loader):
            inputs = ((inputs + 3.0) / 6)
            processed_inputs = processor(images=inputs, return_tensors="pt")
            processed_inputs = processed_inputs.to(device)
            outputs = model.get_image_features(**processed_inputs)
            for j in range(outputs.size()[0]):
                writer.writerow(outputs[j].tolist())
            i += 1
            if i%100 == 0:
                print(f"Completed {i*32} images")

    