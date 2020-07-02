#!/usr/bin/env python
# coding: utf-8

'''
Parts of this code were incorporated from the following github repositories:
1. parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This script has the Encoder and Decoder models and training/validation scripts. 
Edit the parameters sections of this file to specify which models to load/run
''' 

#TODO
# * Increase word embedding dim/attention dim
# * Check tokenizer working well

#DONE
# * Shrink vocab a bit to make it easier... make each word require at least... 
#   10 occurrances to be added...?
# * Remove multi-sentence captions, add commas


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import pickle
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from data_loader_cartoons import get_loader
from nltk.translate.bleu_score import corpus_bleu
from processData import Vocabulary
from tqdm import tqdm
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import skimage.transform
from PIL import Image
import matplotlib.image as mpimg
from torchtext.vocab import Vectors, GloVe
from scipy import misc
from pytorch_pretrained_bert import BertTokenizer, BertModel
import imageio

from networks import Encoder, Decoder

# vocab indices
PAD = 0
START = 1
END = 2
UNK = 3

# loss
class loss_obj(object):
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_loss(imgs, caps, caplens):
    scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
    targets = caps_sorted[:, 1:]
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
    loss = criterion(scores, targets).to(device) + ((1. - alphas.sum(dim=1)) ** 2).mean()
    return loss, decode_lengths

def save_model(tag, epoch, encoder, decoder, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': decoder_optimizer.state_dict(),
        'loss': loss,
        }, f'./checkpoints/decoder_{tag}')

    torch.save({
        'epoch': epoch,
        'model_state_dict': encoder.state_dict(),
        'loss': loss,
        }, f'./checkpoints/encoder_{tag}')

###############
# Train model #
###############

def train():
    print("Started training...")
    for epoch in tqdm(range(num_epochs)):
        decoder.train()
        encoder.train()

        losses = loss_obj()
        num_batches = len(train_loader)

        for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):

            #Replace the very first batch with the longest string
            #if i == 0:
            #    caps[0] = torch.zeros(0)
            #    caplens[0] = 56

            imgs = encoder(imgs.to(device))
            caps = caps.to(device)

            loss, decode_lengths = get_loss(imgs, caps, caplens)

            decoder_optimizer.zero_grad()
            loss.backward()

            # grad_clip decoder
            for group in decoder_optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

            decoder_optimizer.step()

            losses.update(loss.item(), sum(decode_lengths))

            # save model each 100 batches
            if i%5000==0 and i!=0:
                print('epoch '+str(epoch+1)+'/4 ,Batch '+str(i)+'/'+str(num_batches)+' loss:'+str(losses.avg))
                
                 # adjust learning rate (create condition for this)
                for param_group in decoder_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.8

                print('saving model...')
                save_model('mid', epoch, encoder, decoder, loss)
                print('model saved')

            #Attempt for better memory management...
            #del scores, caps_sorted, alphas, targets

        save_model('epoch'+str(epoch+1), epoch, encoder, decoder, loss)
        print('epoch checkpoint saved')

    save_model(model_tag, epoch, encoder, decoder, loss)
    print("Completed training...")  

#################
# Validate model
#################

def print_sample(hypotheses, references, test_references,imgs, alphas, k, show_att, losses):
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))

    print("Validation loss: "+str(losses.avg))
    print("BLEU-1: "+str(bleu_1))
    print("BLEU-2: "+str(bleu_2))
    print("BLEU-3: "+str(bleu_3))
    print("BLEU-4: "+str(bleu_4))

    img_dim = 336 # 14*24
    
    hyp_sentence = []
    for word_idx in hypotheses[k]:
        hyp_sentence.append(vocab.idx2word[word_idx])
    
    ref_sentence = []
    for word_idx in test_references[k]:
        ref_sentence.append(vocab.idx2word[word_idx])

    print('Hypotheses: '+" ".join(hyp_sentence))
    print('References: '+" ".join(ref_sentence))
        
    img = imgs[0][k] 
    imageio.imwrite('img.jpg', img)
  
    if show_att:
        image = Image.open('img.jpg')
        image = image.resize([img_dim, img_dim], Image.LANCZOS)
        for t in range(len(hyp_sentence)):

            plt.subplot(np.ceil(len(hyp_sentence) / 5.), 5, t + 1)

            plt.text(0, 1, '%s' % (hyp_sentence[t]), color='black', backgroundcolor='white', fontsize=12)
            plt.imshow(image)
            current_alpha = alphas[0][t, :].detach().numpy()
            alpha = skimage.transform.resize(current_alpha, [img_dim, img_dim])
            if t == 0:
                plt.imshow(alpha, alpha=0)
            else:
                plt.imshow(alpha, alpha=0.7)
            plt.axis('off')
    else:
        img = imageio.imread('img.jpg')
        plt.imshow(img)
        plt.axis('off')
        plt.show()


def validate():

    references = [] 
    test_references = []
    hypotheses = [] 
    all_imgs = []
    all_alphas = []

    print("Started validation...")
    decoder.eval()
    encoder.eval()

    losses = loss_obj()

    num_batches = len(val_loader)
    # Batches
    for i, (imgs, caps, caplens) in enumerate(tqdm(val_loader)):

        imgs_jpg = imgs.numpy() 
        imgs_jpg = np.swapaxes(np.swapaxes(imgs_jpg, 1, 3), 1, 2)
        
        # Forward prop.
        imgs = encoder(imgs.to(device))
        caps = caps.to(device)

        scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        loss = criterion(scores_packed, targets_packed)
        loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
        losses.update(loss.item(), sum(decode_lengths))

         # References
        for j in range(targets.shape[0]):
            img_caps = targets[j].tolist() # validation dataset only has 1 unique caption per img
            clean_cap = [w for w in img_caps if w not in [PAD, START, END]]  # remove pad, start, and end
            img_captions = list(map(lambda c: clean_cap,img_caps))
            test_references.append(clean_cap)
            references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            pred = p[:decode_lengths[j]]
            pred = [w for w in pred if w not in [PAD, START, END]]
            temp_preds.append(pred)  # remove pads, start, and end
        preds = temp_preds
        hypotheses.extend(preds)
        
        if i == 0:
            all_alphas.append(alphas)
            all_imgs.append(imgs_jpg)

    print("Completed validation...")
    print_sample(hypotheses, references, test_references, all_imgs, all_alphas,1,False, losses)

##############
# Parameters #
##############

parser = argparse.ArgumentParser(description='')
parser.add_argument('model', default = 'baseline',
                            help = "Do not test on simulated data.")
parser.add_argument('--no-train', dest='no_sim', action = 'store_true',
                            help = "Do not test on simulated data.")
parser.add_argument('--no-real', dest='no_real', action = 'store_true',
                            help = "Do not test on real data.")

args = parser.parse_args()

valid_models = ['bert', 'glove', 'baseline']

# hyperparams
grad_clip = 5.
num_epochs = 6
batch_size = 16 
decoder_lr = 0.0004

glove_model = False
bert_model = False

str_models = ' '.join(valid_models)
assert args.model in valid_models, f'Model name not valid, choose one of {str_models}'

if args.model == 'bert':
    bert_model = True 
elif args.model == 'glove':
    glove_model = True 
    
model_tag = args.model 

from_checkpoint = False
train_model = True
valid_model = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
BertModel = BertModel.from_pretrained('bert-base-uncased').to(device)
BertModel.eval()

# Load GloVe
if glove_model:
    glove_vectors = pickle.load(open('glove.6B/glove_words.pkl', 'rb'))
    glove_vectors = torch.tensor(glove_vectors)
else:
    glove_vectors = None

# Load vocabulary
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# load data
train_loader = get_loader('train', vocab, batch_size)
val_loader = get_loader('val', vocab, batch_size)

#############
# Init model
#############

criterion = nn.CrossEntropyLoss().to(device)

if from_checkpoint:

    encoder = Encoder().to(device)
    decoder = Decoder(vocab_size=len(vocab),
                        use_glove=glove_model, 
                        use_bert=bert_model, 
                        device = device,
                        tokenizer = tokenizer,
                        vocab = vocab,
                        bert_model = BertModel, 
                        glove_vectors = glove_vectors).to(device)

    if torch.cuda.is_available():
        encoder_checkpoint = torch.load(f'./checkpoints/encoder_{model_tag}')
        decoder_checkpoint = torch.load(f'./checkpoints/decoder_{model_tag}')
        if bert_model:
            print('Pre-Trained BERT Model')
        elif glove_model:
            print('Pre-Trained GloVe Model')
        else:
            print('Pre-Trained Baseline Model')
    else:
        encoder_checkpoint = torch.load(f'./checkpoints/encoder_{model_tag}', map_location='cpu')
        decoder_checkpoint = torch.load(f'./checkpoints/decoder_{model_tag}', map_location='cpu')
        if bert_model:
            print('Pre-Trained BERT Model')
        elif glove_model:
            print('Pre-Trained GloVe Model')
        else:
            print('Pre-Trained Baseline Model')

    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr)
    decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
    decoder_optimizer.load_state_dict(decoder_checkpoint['optimizer_state_dict'])
else:
    encoder = Encoder().to(device)
    decoder = Decoder(vocab_size=len(vocab),
                        use_glove=glove_model, 
                        use_bert=bert_model, 
                        device = device,
                        tokenizer = tokenizer,
                        vocab = vocab,
                        bert_model = BertModel,
                        glove_vectors = glove_vectors).to(device)
    decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr)

######################
# Run training/validation
######################

if train_model:
    print(f"Training {args.model}")
    train()

if valid_model:
    print(f"Validating {args.model}")
    validate()