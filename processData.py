#!/usr/bin/env python
'''
This code is mainly taken from the following github repositories:
1.  parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script processes the COCO dataset
'''  

import os
import pickle
from collections import Counter
import nltk
from PIL import Image
from pycocotools.coco import COCO

from shutil import copyfile
import pandas as pd
import string 

def tokenize_caption(caption):
    #print(caption)
    #Just ASCII
    encoded_string = caption.encode("ascii", "ignore")
    caption = encoded_string.decode()
    #Remove most punctuation
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    #Tokenize and make lower case
    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
    return tokens

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_caption_vocab(csvs, threshold):
    counter = Counter()
    for csv in csvs:
        df = pd.read_csv(csv, header = None, 
                            names = ['idx', 'file', 'caption'],
                            index_col = 'idx')
        captions = df['caption']
        for caption in captions:
            if type(caption) is not str: continue
            #print(caption)
            tokens = tokenize_caption(caption)
            #print(tokens)
            counter.update(tokens)

    # omit non-frequent words
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>') # 0
    vocab.add_word('<start>') # 1
    vocab.add_word('<end>') # 2
    vocab.add_word('<unk>') # 3

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    # omit non-frequent words
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>') # 0
    vocab.add_word('<start>') # 1
    vocab.add_word('<end>') # 2
    vocab.add_word('<unk>') # 3

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

def process_cartoons(cartoon_path, output_path, train_prop = 0.9, max_words = 30,
                     max_chars = 300):
    dirs = [a for a in os.listdir(cartoon_path) if os.path.isdir(cartoon_path +
             '/' + a)]
    #Split into train and val
    n_train = int(len(dirs)*train_prop)
    train_csv = open(f'./data/train_captions.csv', 'w')
    val_csv = open(f'./data/val_captions.csv', 'w')
    count = 0
    for idx,dr in enumerate(dirs):
        #print(f'Copying {dr}')
        if idx < n_train: split = 'train'
        else: split = 'val'
        #Open captions.txt
        if os.path.exists(f'{cartoon_path}/{dr}/{dr}_captions.txt'):
            fn = f'{cartoon_path}/{dr}/{dr}_captions.txt'
        elif os.path.exists(f'{cartoon_path}/{dr}/{dr}_captions.csv'):
            fn = f'{cartoon_path}/{dr}/{dr}_captions.csv'
        elif os.path.exists(f'{cartoon_path}/{dr}/{dr}_captions_output.csv'):
            fn = f'{cartoon_path}/{dr}/{dr}_captions_output.csv'
        elif os.path.exists(f'{cartoon_path}/{dr}/round1_cardinal/round1_captions.txt'):
            fn = f'{cartoon_path}/{dr}/round1_cardinal/round1_captions.txt'
        elif os.path.exists(f'{cartoon_path}/{dr}/round1_cardinal/captions.txt'):
            fn = f'{cartoon_path}/{dr}/round1_cardinal/captions.txt'
        else:
            fn = f'{cartoon_path}/{dr}/{dr}_captions_output.txt'
        cap_in = open(fn, 'r')
        try:
            for line in cap_in:
                line = line.replace('"', '\'').rstrip()
                #Remove really long lines, or lines with too many words...
                if type(line) is not str: continue
                encoded_string = line.encode("ascii", "ignore")
                line = encoded_string.decode()
                if len(line) > max_chars: continue
                tokens = tokenize_caption(line)
                if len(tokens) > max_words: continue
                if len(tokens) == 0: continue
                caption = f'{count},{dr}.jpg,"{line}"\n'
                if idx < n_train:
                    train_csv.write(caption)
                else:
                    val_csv.write(caption)
                count += 1
        except UnicodeDecodeError:
            print(f"Cannot read {fn}, skipping these captions. Unicode decode error.")
        cap_in.close()

        copyfile(f'{cartoon_path}/{dr}/{dr}.jpg', f'{output_path}_{split}/{dr}.jpg')
    train_csv.close()
    val_csv.close()

def main(csvs,vocab_path,threshold):
    vocab = build_caption_vocab(csvs=csvs, threshold=threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Resizing images...")
    splits = ['val','train']

    for split in splits:
        folder = './data/nycartoons_%s' %split
        resized_folder = './data/nycartoons_%s_resized/' %split
        if not os.path.exists(resized_folder):
            os.makedirs(resized_folder)
        image_files = os.listdir(folder)
        num_images = len(image_files)
        for i, image_file in enumerate(image_files):
            with open(os.path.join(folder, image_file), 'r+b') as f:
                with Image.open(f) as image:
                    image = resize_image(image)
                    image = image.convert("RGB")
                    image.save(os.path.join(resized_folder, image_file), image.format)
    print("Done resizing images.")

cartoon_input_path = '/home/lansdell/projects/caption-contest-data/contests/info/'
cartoon_output_path = './data/nycartoons'
csvs = ['./data/train_captions.csv','./data/val_captions.csv']
#caption_path = './data/annotations/captions_train2014.json'
vocab_path = './data/vocab.pkl'
threshold = 5

if __name__ == "__main__":
    process_cartoons(cartoon_input_path, cartoon_output_path)
    main(csvs,vocab_path,threshold)