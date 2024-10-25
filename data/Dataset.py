from torch.utils.data import Dataset
from torchvision.transforms import Lambda
import torchvision.transforms as transforms
from PIL import Image
import random
import torch

transform = transforms.Compose([
    Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def norm_in_range(lis, epsilon):
  buffer = []
  for i in lis:
    buffer.append(1-i)

  maximum = float(max(buffer))
  new_lis = {}


  for i in lis:
    if maximum == 0:
      new_val = 1.0
    else:
      new_val = epsilon + (1-i)/maximum
    new_lis[i] = new_val
  return new_lis

def generate_train_dataset(dataset, sampling_ratios, baseline):
  new_dataset = []
  for i in range(len(sampling_ratios)):
    rand_val = random.random()
    if baseline:
      new_dataset.append(i)
      continue
    if sampling_ratios[i] < 1:
      if rand_val <= sampling_ratios[i]:
        new_dataset.append(i)
    elif sampling_ratios[i] == 1.0:
      new_dataset.append(i)
    else:
      new_dataset.append(i)
      if rand_val <= (sampling_ratios[i]-1):
        new_dataset.append(i)
  return ImageDatasetSampled(dataset, new_dataset, transform=transform)

class ImageDataset(Dataset):
    def __init__(self, dataset, split, transform):
        self.dataset = dataset[split]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        print(type(image))
        label = self.dataset[idx]['label']
        if self.transform:
            image = self.transform(image)
        print(type(image))
        return image, label
        

def custom_collate_fn(batch):
    
    transform = transforms.Compose([
      Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    labels = []
    images = []
    for item in batch:

        try:
            image, lab = transform(item['image']), item['label']
        except TypeError:
            image, lab = transform(item[0]), item[1]   
        image = torch.squeeze(image) 
        image = torch.squeeze(image) 
        images.append(image)
        labels.append(lab)
        
    # print(labels)

    try:
        return torch.stack(images, dim=0), torch.Tensor(labels)
    except ValueError:
        labels = torch.stack(labels, dim=0)
        return torch.stack(images, dim=0), labels

    

class ImageDatasetSampled(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # This part might need adjustment based on the dataset's structure
        image = self.dataset[self.indices[idx]]['image']
        label = self.dataset[self.indices[idx]]['label']
        if self.transform:
            image = self.transform(image)
        return image, label