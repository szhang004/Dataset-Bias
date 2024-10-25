import os
import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import zipfile
import requests
from io import BytesIO
from sklearn import preprocessing
import keras
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image
from data.Dataset import ImageDataset

BATCH_SIZE = 20
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
TRAINING_IMAGES_DIR = '/project/aortega_421/skzhang/unsupervised-bias-detection-master/tiny-imagenet-200/train'
TRAIN_SIZE = NUM_IMAGES

NUM_VAL_IMAGES = 10000
VAL_IMAGES_DIR = '/project/aortega_421/skzhang/unsupervised-bias-detection-master/tiny-imagenet-200/val'

IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS
IMAGES_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

def download_images(url):
    if (os.path.isdir(TRAINING_IMAGES_DIR)):
        print ('Images already downloaded...')
        return

    r = requests.get(url, stream=True)
    print ('Downloading ' + url )
    zip_ref = zipfile.ZipFile(BytesIO(r.content))
    zip_ref.extractall('./')
    zip_ref.close()


from torch.utils.data import Dataset
class TrainImageDataset(Dataset):
  def __init__(self):
    self.image_dir = TRAINING_IMAGES_DIR
    self.dirs = os.listdir(self.image_dir)
    print(self.dirs)

  def __getitem__(self, idx):
    print(idx)
    labels = []
    names = []
    bboxes = []
    images = []
    bbox_file = open(os.path.join(self.image_dir, self.dirs[idx], self.dirs[idx] + '_boxes.txt'))
    bbox_data = {}
    for row in bbox_file:
      line = row.split()
      bbox_data[line[0]] = [int(line[1]), int(line[2]), int(line[3]), int(line[4])]

    for image in os.listdir(self.image_dir + '/' + self.dirs[idx] + '/images/'):
      image_file = os.path.join(self.image_dir, self.dirs[idx] + '/images/', image)

      # reading the images as they are; no normalization, no color editing
      image_data = mpimg.imread(image_file)
      #print ('Loaded Image', image_file, image_data.shape)
      if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
          labels.append(self.dirs[idx])
          names.append(image)
          bboxes.append(bbox_data[image])
          images.append(image_data.flatten())
    return images, labels, bboxes


  def __len__(self):
    return 200

def get_label_from_name(data, name):
    for idx, row in data.iterrows():
        if (row['File'] == name):
            return row['Class']

    return None


def load_validation_images(testdir, validation_data, batch_size=NUM_VAL_IMAGES):
    labels = []
    names = []

    image_index = 0
    val_data = pd.read_csv(VAL_IMAGES_DIR + '/val_annotations.txt', sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'W', 'H'])
    annotations = {}
    for i in range(len(val_data)):
      annotations[val_data['File'][i]] = [val_data['X'][i], val_data['Y'][i], val_data['W'][i], val_data['H'][i]]
    images = np.ndarray(shape=(batch_size, IMAGE_ARR_SIZE))
    val_images = os.listdir(testdir + '/images/')

    # Loop through all the images of a val directory
    batch_index = 0
    bboxes = []

    for image in val_images:
        image_file = os.path.join(testdir, 'images/', image)
        #print (testdir, image_file)
        print("hi")
        # reading the images as they are; no normalization, no color editing
        image_data = mpimg.imread(image_file)
        if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
            images[image_index, :] = image_data.flatten()
            image_index += 1
            labels.append(get_label_from_name(validation_data, image))
            names.append(image)
            bboxes.append(annotations[image])
            batch_index += 1

        if (batch_index >= batch_size):
            break

    print ("Loaded Validation images ", image_index)
    return (images, np.asarray(labels), np.asarray(names), np.asarray(bboxes))

def plot_object(data):
    plt.figure(figsize=(1,1))
    image = data.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def get_dataset_imagenet(path):
    download_images(IMAGES_URL)
    train_data = TrainImageDataset()
    print(train_data)
    loader = DataLoader(train_data, batch_size=1)
    training_images = []
    training_labels = []
    bbox_data_train = []
    for (train_img, labels, bboxes) in iter(loader):
        training_images.append(train_img)
        training_labels.append(labels)
        bbox_data_train.append(bboxes)

    label_dict = {}
    count_dict = {}
    count_dict["train"] = {}
    count_dict["val"] = {}
    unique_count = 0
    for i in range(len(training_labels)):
        for j in range(len(training_labels[i])):
            if training_labels[i][j][0] not in label_dict.keys():
                label_dict[training_labels[i][j][0]] = unique_count
                unique_count+=1
    train_images_resized = []
    train_labels_encoded = []

    val_images_resized = []
    val_labels_encoded = []

    bbox_data = {}
    for i in range(len(training_labels)):
        for j in range(len(training_labels[i])):
            train_images_resized.append(transforms.ToPILImage()(np.transpose(np.reshape(training_images[i][j], (64, 64, 3)), (2, 0, 1))))
            train_labels_encoded.append(label_dict[training_labels[i][j][0]])
    bbox_data["train"] = bbox_data_train

    #val_data = pd.read_csv(VAL_IMAGES_DIR + '/val_annotations.txt', sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'W', 'H'])
    #val_images, val_labels, val_files, bbox_data_val = load_validation_images(VAL_IMAGES_DIR, val_data, batch_size=NUM_VAL_IMAGES)
    #for i in range(len(val_labels)):
    #    val_images_resized.append(Image.fromarray(np.uint8(np.reshape(val_images[i], (64, 64, 3)))))
    #    val_labels_encoded.append(label_dict[val_labels[i]])
    #bbox_data["val"] = bbox_data_val

    for i in range(len(train_labels_encoded)):
        if train_labels_encoded[i] in count_dict["train"].keys():
            count_dict["train"][train_labels_encoded[i]]+=1
        else:
            count_dict["train"][train_labels_encoded[i]] = 1
    
    #for i in range(len(val_labels_encoded)):
    #    if val_labels_encoded[i] in count_dict["val"].keys():
    #        count_dict["val"][val_labels_encoded[i]]+=1
    #    else:
    #        count_dict["val"][val_labels_encoded[i]] = 1
    dataset = {}
    dataset["train"] = []
    #dataset["val"] = []
    for i in range(len(train_images_resized)):
        dataset["train"].append({})
        dataset["train"][i]["image"] = train_images_resized[i]
        dataset["train"][i]["label"] = train_labels_encoded[i]
    
    #for i in range(len(val_images_resized)):
    #    dataset["val"].append({})
    #    dataset["val"][i]["image"] = val_images_resized[i]
    #    dataset["val"][i]["label"] = val_labels_encoded[i]
    

    return dataset, bbox_data, count_dict