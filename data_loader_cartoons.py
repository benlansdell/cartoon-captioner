'''
This code is mainly taken from the following github repositories:
1.  parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script loads the COCO dataset in batches to be used for training/testing
''' 

import os
import nltk
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
import pandas as pd
import pickle 
from processData import Vocabulary

class DataLoader(data.Dataset):
    def __init__(self, root, csv, vocab, transform=None):

        self.root = root
        df = pd.read_csv(csv, header = None, names = ['idx', 'file', 'caption'],
                            index_col = 'idx')
        self.data_dict = df.to_dict('index')
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        vocab = self.vocab
        path = self.data_dict[index]['file']
        caption = self.data_dict[index]['caption']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.data_dict)

def collate_fn(data):
    data.sort(key=lambda  x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def get_loader(method, vocab, batch_size):

    # train/validation paths
    if method == 'train':
        root = 'data/nycartoons_train_resized'
        csv = 'data/train_captions.csv'
    elif method =='val':
        root = 'data/nycartoons_val_resized'
        csv = 'data/val_captions.csv'

    # resnet transformation/normalization
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])

    nycc = DataLoader(root=root, csv=csv, vocab=vocab, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=nycc,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=1,
                                              collate_fn=collate_fn)
    return data_loader

#Test dataloader
#BS = 64
#with open('data/vocab.pkl', 'rb') as f:
#    vocab = pickle.load(f)
#loader = get_loader('train', vocab, BS)
#images, captions, lengths = next(iter(loader))