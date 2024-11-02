import os
import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pytorch as torch
import zipfile
import requests
from io import BytesIO
from sklearn import preprocessing
import keras
from torch.utils.data import DataLoader
import csv
import time

import argparse
from data.celebA import get_dataset_celebA
from data.cifar100 import get_dataset_cifar
from data.cifar100c import get_dataset_cifarc
from data.imagenet import get_dataset_imagenet
from data.tinyimagenetc import get_dataset_tinyimagenet_c
from train import imagenetroot, cifarroot, celebAroot, MODEL_DIR, CLUSTERS_DIR
from utils.nnk import eval_cluster, NNK_means
from utils.features import contrast, color_distrib, bbox_area
from data.Dataset import ImageDataset, transform, custom_collate_fn

seed = 145
metric = 'error' # anomaly detection metric

root = "/project/aortega_421/skzhang/unsupervised-bias-detection-master/"
featuresroot = "/project/aortega_421/skzhang/unsupervised-bias-detection-master/features"

def generate_features(training_data, val_data, features, ds):

  train_vecs = []

  fts = ["contrast", "color", "size", "CLIP"]
  if all(ft not in features for ft in fts):
      raise("Invalid feature(s)- please select atleast 1 out of [CLIP, contrast, color, size]")
  
  if "contrast" in features:
    if not os.path.exists(featuresroot + "/" + ds + "_contrast_t"):
      contrast_results_t = np.zeros((len(training_data),1))
      contrast_results_v = np.zeros((len(val_data),1))

      for i in range(len(val_data)):
        contrast_results_t[i] = contrast(np.array(training_data[i]))
      max_contrast_t = max(contrast_results_t)
      for i in range(len(contrast_results_t)):
        contrast_results_t[i] = contrast_results_t[i]/max_contrast_t

      for i in range(len(val_data)):
        contrast_results_v[i] = contrast(np.array(val_data[i]))
      max_contrast_v = max(contrast_results_v)
      for i in range(len(contrast_results_v)):
        contrast_results_v[i] = contrast_results_v[i]/max_contrast_v

      np.savetxt(featuresroot + "/" + ds + "_contrast_t", contrast_results_t, delimiter=",")
      np.savetxt(featuresroot + "/" + ds + "_contrast_v", contrast_results_v, delimiter=",")
    else:
      contrast_results_t = np.loadtxt(featuresroot + "/" + ds + "_contrast_t", delimiter=",")
      contrast_results_v = np.loadtxt(featuresroot + "/" + ds + "_contrast_v", delimiter=",")
    train_vecs = np.concatenate((train_vecs, contrast_results_t), axis=1)
    val_vecs = np.concatenate((val_vecs, contrast_results_v), axis=1)

  if "color" in features:
    if os.path.exists(featuresroot + "/" + ds + "_color_t"):
      color_results_t = np.zeros((len(training_data),1))
      color_results_v = np.zeros((len(val_data),11))

      for i in range(len(val_data)):
        contrast_results_t[i] = contrast(np.array(training_data[i]))
      max_contrast_t = max(contrast_results_t)
      for i in range(len(contrast_results_t)):
        contrast_results_t[i] = contrast_results_t[i]/max_contrast_t

      for i in range(len(val_data)):
        contrast_results_v[i] = contrast(np.array(val_data[i]))
      max_contrast_v = max(contrast_results_v)
      for i in range(len(contrast_results_v)):
        contrast_results_v[i] = contrast_results_v[i]/max_contrast_v

      np.savetxt(featuresroot + "/" + ds + "_color_t", color_results_t, delimiter=",")
      np.savetxt(featuresroot + "/" + ds + "_color_v", color_results_v, delimiter=",")
    else:
      color_results_t = np.loadtxt(featuresroot + "/" + ds + "_color_t", delimiter=",")
      color_results_v = np.loadtxt(featuresroot + "/" + ds + "_color_v", delimiter=",")
    train_vecs = np.concatenate((train_vecs, color_results_t), axis=1)
    val_vecs = np.concatenate((val_vecs, color_results_v), axis=1)

  if "size" in features and ds == "imagenet":
    if os.path.exists(featuresroot + "/" + ds + "_size_t"):
      if os.path.exists(featuresroot + "/" + ds + "_size_t"):
        bbox_results_t = np.zeros((len(training_data),1))
        bbox_results_v = np.zeros((len(val_data),11))

      for i in range(len(val_data)):
        bbox_results_t[i] = bbox_area(bbox_data["val"][i])
      max_size_t = max(bbox_results_t)
      for i in range(len(bbox_results_t)):
        bbox_results_t[i] = bbox_results_t[i]/max_size_t

      for i in range(len(val_data)):
        bbox_results_v[i] = bbox_area(bbox_data["val"][i])
      max_size_v = max(bbox_results_v)
      for i in range(len(bbox_results_v)):
        bbox_results_v[i] = bbox_results_v[i]/max_size_v

      np.savetxt(featuresroot + "/" + ds + "_size_t", bbox_results_t, delimiter=",")
      np.savetxt(featuresroot + "/" + ds + "_size_v", bbox_results_v, delimiter=",")
    else:
      bbox_results_t = np.loadtxt(featuresroot + "/" + ds + "_size_t", delimiter=",")
      bbox_results_v = np.loadtxt(featuresroot + "/" + ds + "_size_v", delimiter=",")
    train_vecs = np.concatenate((train_vecs, bbox_results_t), axis=1)
    val_vecs = np.concatenate((val_vecs, bbox_results_v), axis=1)

  return train_vecs, val_vecs
  

def nnk_cluster(train_vecs, training_data, count_dict, training_labels, epochs, atoms, top_k, ep):
  start_time = time.time()

  start = 0
  end = len(training_data) 
  
  try:
    os.remove(CLUSTERS_DIR + "/" + training_data + "assignments.csv")
  except OSError:
    pass
  with open(CLUSTERS_DIR + "/" + training_data + "assignments.csv", 'w+', newline='') as csvfile:
    writer = csv.writer(csvfile)
    while end < 100000:
      print("Clustering on class", idx, "...")
      data = train_vecs[start:end+1]
      data = data.astype(np.float32)
      nnkmodel = NNK_means(num_epochs=epochs, metric=metric, n_components=atoms, top_k=top_k, ep=ep, weighted=False, num_warmup=2)
      nnkmodel.fit(data)

      assignments = [[] for i in range(atoms)]
      assignment_indices = [[] for i in range(atoms)]
      codes = nnkmodel.get_codes(data)
      for i in range(end-start+1):
        max_idx = torch.argmax(codes[i])
        obj = {}
        obj['image'] = training_data[start+i]
        obj['label'] = training_labels[start+i]
        assignments[max_idx].append(obj)
        assignment_indices[max_idx].append(start+i)

      cluster_accuracies = [[] for i in range(atoms)]
      for i in range(len(assignments)):
          if len(assignment_indices[i]) > 0:
              acc = eval_cluster(assignment_indices[i])
          else:
              acc = np.nan
          cluster_accuracies[i] = acc
          print(f'Cluster {i}: Accuracy = {acc * 100:.2f}%')
      dict = {}
      for i in range(atoms):
        for j in range(len(assignment_indices[i])):
          dict[assignment_indices[i][j]] = (cluster_accuracies[i], i)
      for i in range(start, end+1):
        writer.writerow([i, dict[i][0], dict[i][1]])

      end_time = time.time()
      execution_time = end_time - start_time

      print(f"Execution time: {execution_time} seconds")


      start += count_dict[idx]
      idx += 1
      if idx < 200:
        end += count_dict[idx]
      else:
        break

def get_dataset(args):
  if args.dataset_v == "imagenet": return get_dataset_imagenet(imagenetroot)
  elif args.dataset_v == "cifar100": return get_dataset_cifar(cifarroot)
  elif args.dataset_v == "celebA": return get_dataset_celebA(celebAroot)
  else: ValueError("Invalid Dataset")


if __name__ == "__main__":
    
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, default="imagenet")
  parser.add_argument("--cset", type=str, default="train")
  parser.add_argument("--features", type=list, default=["CLIP"]) # CLIP, contrast, color, size
  parser.add_argument("--cset", type=str, default="train")
  parser.add_argument("--Random", type=bool, default=False)
  parser.add_argument("--atoms", type=int, default=50)
  parser.add_argument("--sparsity", type=int, default=15)
  parser.add_argument("--baseline", type=bool, default=False)
  parser.add_argument("--base", type=str, default="resnet-s")
  parser.add_argument("--epochs", type=int, default=7)
  parser.add_argument("--entropy", type=int, default=0)
  args = parser.parse_args()

  dataset, bbox_data, counts = get_dataset(args)
  train_vecs, val_vecs = generate_features(dataset["train"], dataset["val"], args.features, args.dataset)

  if len(train_vecs) > 0 :
    nnk_cluster(train_vecs, dataset["label"], counts, args.dataset, args.epochs, args.atoms, args.sparsity, args.entropy, CLUSTERS_DIR)
  if ("CLIP" in args.features):
    print("hi")
    
`