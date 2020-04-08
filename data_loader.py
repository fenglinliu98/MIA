import json
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import string
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from coco.pycocotools.coco import COCO
import pandas as pd
import base64
import csv
import sys
import zlib
import time
import mmap

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, concepts, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO( json )
        self.ids = list( self.coco.anns.keys() )
        self.vocab = vocab
        self.transform = transform
        self.concepts = concepts

    def __getitem__(self, index):
        """Returns one data pair ( image, caption, image_id, concept )."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        filename = coco.loadImgs(img_id)[0]['file_name']

        if 'val2014' in filename.lower():
            path = 'val2014/' + filename
        elif 'train2014' in filename.lower():
            path = 'train2014/' + filename
        else:
            path = 'test2014/' + filename
            
        image = Image.open( os.path.join( self.root, path ) ).convert('RGB')
        if self.transform is not None:
            image = self.transform( image )

        # Convert caption (string) to word ids.
        tokens = str( caption ).lower().translate( None, string.punctuation ).strip().split()
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        
        # Load image concepts
        image_concepts = []
        for concept in self.concepts:
            if concept['image_id'] == img_id:
                image_concept = concept['image_concepts'] 
                image_concepts.extend([vocab(token) for token in image_concept])
                break
        image_concepts = torch.Tensor( image_concepts )
        
        return image, target, img_id, filename, image_concepts

    def __len__(self):
        return len( self.ids )

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
        img_ids: image ids in COCO dataset, for evaluation purpose
        filenames: image filenames in COCO dataset, for evaluation purpose
    """

    # Sort a data list by caption length (descending order).
    data.sort( key=lambda x: len( x[1] ), reverse=True )
    images, captions, img_ids, filenames, concepts = zip( *data ) # unzip

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    img_ids = list( img_ids )
    filenames = list( filenames )

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max( lengths ) ).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]    
        
    # Merge image_concepts (from tuple of 1D tensor to 2D tensor).
    length_concepts = len(concepts[0])
    image_concepts = torch.zeros(len(concepts), length_concepts).long()
    for j, capj in enumerate(concepts):
        end_concept = length_concepts
        image_concepts[j, :end_concept] = capj[:end_concept]
         
    return images, targets, lengths, img_ids, filenames, image_concepts

def get_loader( root, json, concepts, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       concepts=concepts,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
