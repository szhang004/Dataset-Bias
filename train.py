import argparse
import os 
import random
from torch.utils.data import DataLoader
from data.Dataset import ImageDataset, ImageDatasetSampled, transform
import csv
# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet50
# from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt
import time, os, copy, numpy as np
from livelossplot import PlotLosses
import sys

from data.celebA import get_dataset_celebA
from data.cifar100 import get_dataset_cifar
from data.imagenet import get_dataset_imagenet

imagenetroot = "/project/aortega_421/skzhang/unsupervised-bias-detection-master/tiny-imagenet-200"
cifarroot = "/project/aortega_421/skzhang/unsupervised-bias-detection-master/cifar100"
celebAroot = "/project/aortega_421/skzhang/unsupervised-bias-detection-master/celebA"
CLUSTERS_DIR = "/project/aortega_421/skzhang/unsupervised-bias-detection-master/clusters"
CLIP_DIR = "/project/aortega_421/skzhang/unsupervised-bias-detection-master/clip"
MODEL_DIR = "/project/aortega_421/skzhang/unsupervised-bias-detection-master/models"

def norm_in_range(lis, args):
  buffer = []
  for i in lis:
    buffer.append(1-i)
  maximum = float(max(buffer))
  new_lis = {}
  for i in lis:
    if maximum == 0:
      new_val = 1.0
    else:
      new_val = args.epsilon + (1-i)/maximum
    new_lis[i] = new_val
  return new_lis


def generate_train_dataset(dataset, sampling_ratios, args):
  new_dataset = []
  for i in range(len(sampling_ratios)):
    rand_val = random.random()
    if args.baseline:
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
  return ImageDataset(dataset, new_dataset, transform=transform)

def get_dataset(args):
    if args.dataset == "imagenet": return get_dataset_imagenet(imagenetroot)
    elif args.dataset == "cifar100": return get_dataset_cifar(cifarroot)
    elif args.dataset == "celebA": return get_dataset_celebA(celebAroot)
    else: raise("Invalid Dataset")

def get_sampling_ratios(args, dataset, count_dict):
    feat_name = " ".join(args.features)
    if args.Random: feat_name = "random"
    fname = os.path.join(CLUSTERS_DIR, f"assignments_{args.cset}_{args.model}_{args.atoms}_{args.sparsity}_{feat_name}_{args.dataset}.csv")
    if args.baseline: fname = "assignments_a15_s5_scratch.csv"
    
    with open(os.path.join(os.getcwd(),fname), 'r') as csvfile:
        sampling_ratios = []
        all_accuracies = []
        accs = []

        cidx_to_accuracy = {}
        
        cidx = 0
        row_count = 0

        reader = csv.reader(csvfile)
        
        for row in reader:
            if args.baseline:
                for i in range(len(dataset["train"]["images"])): sampling_ratios.append(1)
                break
            cidx_to_accuracy[int(row[2])] = float(row[1])
            accs.append(float(row[1]))
            all_accuracies.append(float(row[1]))
            if row_count%count_dict["train"][cidx] == count_dict["train"][cidx]-1:
                accuracy_map = norm_in_range([cidx_to_accuracy[key] for key in cidx_to_accuracy.keys()], args.epsilon)
                for i in accs:
                    sampling_ratios.append(accuracy_map[i])
                cidx_to_accuracy = {}
                accs = []
                row_count = 0
                cidx += 1
                continue
            row_count += 1
    return sampling_ratios, feat_name

def get_initial_loaders(dataset, sampling_ratios):
    ds = dataset["train"]["images"]
    ds_l = dataset["train"]["labels"]
    new_ds = []
    new_s_r = []
    for i in range(len(ds)):
        obj = {}
        obj['image'] = ds[i]
        obj['label'] = ds_l[i]
        new_ds.append(obj)
        new_s_r.append(sampling_ratios[i])
    dummy = {}
    dummy['train'] = new_ds
    ds = dataset["val"]["images"]
    ds_l = dataset["val"]["labels"]
    new_ds = []
    for i in range(len(ds)):
        obj = {}
        obj['image'] = ds[i]
        obj['label'] = ds_l[i]
        new_ds.append(obj)
    dummy['valid'] = new_ds
    train_ds = ImageDataset(dummy, 'train', transform=transform)
    valid_ds = ImageDataset(dummy, 'valid', transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size =32, shuffle = True)
    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['valid'] = valid_loader
    dataset_sizes = {}
    dataset_sizes['train'] = len(train_ds)
    dataset_sizes['valid'] = len(valid_ds)
    return dataloaders, dataset_sizes
   
def get_model(args):
    if args.model == "deit":
        model = vit_b_16(weights=ViT_B_16_Weights)
    #model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224", attn_implementation="sdpa", torch_dtype=torch.float32)
        if args.dataset == "imagenet":
            model.heads.head = torch.nn.Linear(in_features=768, out_features=200, bias=True)
        elif args.dataset == "cifar100":
            model.heads.head = torch.nn.Linear(in_features=768, out_features=100, bias=True)
        elif args.dataset == "celebA":
            model.heads.head = torch.nn.Linear(in_features=768, out_features=2, bias=True)
        else:
            raise("Invalid dataset")
        
    # Using pretrained weights:
    else:
        model = resnet50(pretrained=True)
        if args.dataset == "imagenet":
            model.fc = torch.nn.Linear(in_features=2048, out_features=200, bias=True)
        elif args.dataset == "cifar100":
            model.fc = torch.nn.Linear(in_features=2048, out_features=100, bias=True)
        elif args.dataset == "celebA":
           model.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
        else: 
           raise("Invalid dataset")
    return model

def train_model(model, dataloaders, dataset, dataset_sizes, criterion, optimizer, scheduler, sampling_ratios, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()
    liveloss = PlotLosses()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
                epoch_train_ds = generate_train_dataset(dataset["train"], sampling_ratios)
                dataset_sizes['train'] = len(epoch_train_ds)
                epoch_train_loader = DataLoader(epoch_train_ds, batch_size=32, shuffle=True)
                dataloaders['train'] = epoch_train_loader
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i,(inputs, labels) in enumerate(dataloaders[phase]):
                # wrap inputs and labels in Variabl
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if args.model == "deit": outputs = outputs.logits
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(dataloaders[phase]), loss.item() * inputs.size(0)), end="")
                sys.stdout.flush()


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        liveloss.update({
            'log loss': avg_loss,
            'val_log loss': val_loss,
            'accuracy': t_acc,
            'val_accuracy': val_acc
        })

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        print(  'Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        print('Best Val Accuracy: {}'.format(best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--model", type=str, default="resnet-s")
    parser.add_argument("--features", type=list, default=["CLIP"])
    parser.add_argument("--cset", type=str, default="train")
    parser.add_argument("--Random", type=bool, default=False)
    parser.add_argument("--atoms", type=int, default=50)
    parser.add_argument("--sparsity", type=int, default=15)
    parser.add_argument("--baseline", type=bool, default=False)
    parser.add_argument("--base", type=str, default="resnet-s")
    parser.add_argument("--epochs", type=int, default=7)

    args = parser.parse_args()

    dataset, bbox_data, counts = get_dataset(args) 
    sampling_ratios, feat_name = get_sampling_ratios(args, dataset, counts)
    loaders, sizes = get_initial_loaders(args, sampling_ratios)
    model = get_model(args)

    
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model = train_model(model, loaders, sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=7)
    
    if args.baseline: torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"baseline-{args.dataset}-{args.model}-finetuned.pth"))
    else: torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{args.epsilon}-{args.atoms}-{args.sparsity}-{args.base}-{args.model}-{args.dataset}-{feat_name}.pth"))
    