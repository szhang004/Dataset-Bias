import argparse
from data.celebA import get_dataset_celebA
from data.cifar100 import get_dataset_cifar
from data.cifar100c import get_dataset_cifarc
from data.imagenet import get_dataset_imagenet
from data.tinyimagenetc import get_dataset_tinyimagenet_c
from train import imagenetroot, cifarroot, celebAroot, MODEL_DIR, CLUSTERS_DIR
from utils.nnk import eval_cluster  
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
import torch
import os
import csv
import numpy as np
from data.Dataset import ImageDataset, transform
from torch.utils.data import DataLoader


tinyimagenetcroot = "/project/aortega_421/skzhang/unsupervised-bias-detection-master/tinyimagenetc"
cifarcroot = "/project/aortega_421/skzhang/unsupervised-bias-detection-master/cifar100c"

def get_dataset(args):
    if args.dataset_v == "imagenet": return get_dataset_imagenet(imagenetroot)
    elif args.dataset_v == "cifar100": return get_dataset_cifar(cifarroot)
    elif args.dataset_v == "cifar100c": return get_dataset_cifarc(cifarcroot, args.ctype)
    elif args.dataset_v == "celebA": return get_dataset_celebA(celebAroot)
    elif args.dataset_v == "tinyimagenetc": return get_dataset_tinyimagenet_c(tinyimagenetcroot, args.ctype, args.cdeg)
    else: ValueError("Invalid Dataset")


def get_model(args):
    feat_name = " ".join(args.features_t)
    if args.Random_t: feat_name = "random"

    if args.model == "deit":
        model = vit_b_16(weights=ViT_B_16_Weights)
    #model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224", attn_implementation="sdpa", torch_dtype=torch.float32)
        if args.dataset_t == "imagenet":
            model.heads.head = torch.nn.Linear(in_features=768, out_features=200, bias=True)
        elif args.dataset_t == "cifar100":
            model.heads.head = torch.nn.Linear(in_features=768, out_features=100, bias=True)
        elif args.dataset_t == "celebA":
            model.heads.head = torch.nn.Linear(in_features=768, out_features=2, bias=True)
        else:
            raise("Invalid dataset")
        
    # Using pretrained weights:
    else:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if args.dataset_t == "imagenet":
            model.fc = torch.nn.Linear(in_features=2048, out_features=200, bias=True)
        elif args.dataset_t == "cifar100":
            model.fc = torch.nn.Linear(in_features=2048, out_features=100, bias=True)
        elif args.dataset_t == "celebA":
           model.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
        else: 
           raise("Invalid dataset")
    
    if args.baseline_t: model_path = os.path.join(MODEL_DIR, f"baseline-{args.dataset_t}-{args.model}-finetuned.pth")
    else: model_path = os.path.join(MODEL_DIR, f"{args.epsilon}-{args.atoms_t}-{args.sparsity_t}-{args.base_t}-{args.model}-{args.dataset_t}-{feat_name}.pth")
    model.load_state_dict(torch.load(model_path)) 

    
    return model

def get_clusters(args, count_dict):
    if args.Random_t: feat_name = "random"
    
    fname = f"assignments_{args.cset}_{args.base_v}_{args.atoms_v}_{args.sparsity_v}_{feat_name}_{args.dataset_v}.csv"
    if os.path.exists(os.path.join(CLUSTERS_DIR, fname)):
        with open(os.path.join(CLUSTERS_DIR,fname), 'r') as csvfile:
            reader = csv.reader(csvfile)
            row_count = 0
            clusters_all = []
            clusters_one = [[] for i in range(args.atoms_v)]
            for row in reader:
                if row_count%count_dict["val"][cidx] == count_dict[cidx]["val"]-1:
                    clusters_all.append(clusters_one)
                    clusters_one = [[] for i in range(args.atoms_v)]
                    cidx += 1
                    row_count = 0
                clusters_one[int(row[2])].append(int(row[0]))
                row_count += 1
            clusters_all.append(clusters_one)
    else:
        print("No clusters file found: performing default evaluation")
        return None
    return clusters_all


def test_model(args, dataset, model, count_dict, clusters):
    if clusters:
        all_vars = []
        running_acc = 0
        running_tot = 0
        val_images_resized = dataset["val"]["images"]
        val_labels_encoded = dataset["val"]["labels"]
        for clus in clusters:
            accs = []
            for c in clus:
                if len(c) > 0:
                    a, t = eval_cluster(model, c, val_images_resized, val_labels_encoded, False)
                    running_acc += a*t
                    running_tot += t
                    accs.append(a)
            v = np.var(accs)
            if v:
                all_vars.append(v)

        print("Mean:", sum(all_vars)/len(all_vars))
        print("Acc:", running_acc/running_tot)
    
    else:
        model.eval()
        model.cuda()
        dummy = {}
        ds = dataset["val"]["images"]
        ds_l = dataset["val"]["labels"]
        new_ds = []
        for i in range(len(ds)):
            obj = {}
            obj['image'] = ds[i]
            obj['label'] = ds_l[i]
            new_ds.append(obj)
        dummy['valid'] = new_ds
        valid_ds = ImageDataset(dummy, 'valid', transform=transform)
        valid_loader = DataLoader(valid_ds, batch_size =32, shuffle = True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for images, labels in iter(valid_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        print("Acc:", accuracy)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_t", type=str, default="imagenet")
    parser.add_argument("--model", type=str, default="resnet-s")
    parser.add_argument("--features_t", type=list, default=["CLIP"])
    parser.add_argument("--cset", type=str, default="val")
    parser.add_argument("--Random_t", type=bool, default=False)
    parser.add_argument("--atoms_t", type=int, default=50)
    parser.add_argument("--sparsity_t", type=int, default=15)
    parser.add_argument("--baseline_t", type=bool, default=False)
    parser.add_argument("--base_t", type=str, default="resnet-s")
    parser.add_argument("--epochs_t", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=0)

    parser.add_argument("--dataset_v", type=str, default="imagenet")
    parser.add_argument("--Random_v", type=bool, default=False)
    parser.add_argument("--atoms_v", type=int, default=15)
    parser.add_argument("--sparsity_v", type=int, default=5)
    parser.add_argument("--features_v", type=list, default=["CLIP"])
    parser.add_argument("--base_v", type=str, default="resnet-s")
    
    parser.add_argument("--ctype", type=str, default="brightness")
    parser.add_argument("--cdeg", type=str, default="1")
    
    args = parser.parse_args()
    dataset, bbox_data, counts = get_dataset(args) 
    model = get_model(args) 
    clusters = get_clusters(args, counts)
    test_model(args, dataset, model, counts, clusters)


    





