import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import glob
import pickle

from build_vocab import Vocabulary
from Models.Generator import Generator

from torch.autograd import Variable
from torchvision import transforms, datasets
from coco.pycocotools.coco import COCO
from coco.pycocoevalcap.eval import COCOEvalCap

import matplotlib.pyplot as plt

# Variable wrapper
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable( x, volatile=volatile )

# MS COCO evaluation data loader
class CocoEvalLoader( datasets.ImageFolder ):
    def __init__( self, root, ann_path, concept_path, vocab, 
                 transform=None, loader=datasets.folder.default_loader ):
        '''
        Customized COCO loader to get Image ids and Image Filenames
        root: path for images
        ann_path: path for the annotation file (e.g., caption_val2014.json)
        '''
        self.root = root
        self.transform = transform
        self.loader = loader
        self.imgs = json.load( open( ann_path, 'r' ) )['images']
        self.image_concept = json.load(open( concept_path , 'r'))
        self.vocab = vocab
        
    def __getitem__(self, index):

        filename = self.imgs[ index ]['file_name']
        img_id = self.imgs[ index ]['id']
        
        # Filename for the image
        if 'val2014' in filename.lower():
            path = os.path.join( self.root, 'val2014' , filename )
        elif 'train2014' in filename.lower():
            path = os.path.join( self.root, 'train2014' , filename )
        else:
            path = os.path.join( self.root, 'test2014', filename )

        img = self.loader( path )
        if self.transform is not None:
            img = self.transform( img )
         
        # Load the image concepts
        image_concepts = []
        for concept in self.image_concept:
            if concept['image_id'] == img_id:
                current_concept = concept['image_concepts']
                image_concepts.extend([self.vocab(token) for token in current_concept])
                break

        image_concepts = torch.LongTensor( image_concepts )
        
        return img, image_concepts, img_id, filename

    def __len__(self):
        return len( self.imgs )
        
# MSCOCO Evaluation function
def main( args ):

    '''
    model: trained model to be evaluated
    args: parameters
    '''
    # Load vocabulary wrapper.
    with open( args.vocab_path, 'rb') as f:
        vocab = pickle.load( f )
    
    # Create results directory 
    if not os.path.exists( os.path.join( args.result_path, args.basic_model ) ):
        os.makedirs( os.path.join( args.result_path, args.basic_model ) )
    
    # List and sort all checkpoints in the storage directory
    if args.use_MIA:
        checkpoint_dir = os.path.join( args.save_dir_path, args.basic_model + "-MIA" )
    else:
        checkpoint_dir = os.path.join( args.save_dir_path, args.basic_model )
        
    checkpoint_list = os.listdir( checkpoint_dir )
    checkpoint_list.sort()

    # Load Caption Model
    for checkpoint in checkpoint_list:
        checkpoint_path = os.path.join( checkpoint_dir, checkpoint )
        Caption_Generator = Generator( args, checkpoint_path, len( vocab ) )

        transform = transforms.Compose([
            transforms.Resize( ( args.crop_size, args.crop_size ) ),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

        # Wrapper the COCO VAL dataset
        eval_data_loader = torch.utils.data.DataLoader(
            CocoEvalLoader( args.image_dir, args.caption_test_path, args.concept_path, vocab, transform ),
            batch_size = args.eval_batch_size,
            shuffle = False, num_workers = args.num_workers,
            drop_last = False )
        
        epoch = int( checkpoint.split('-')[1].split('.')[0] )
        
        # Generated captions to be compared with GT
        results = []
        
        print '---------------------Start evaluation on MS-COCO dataset-----------------------'
        for i, ( images, image_concepts, image_ids, _ ) in enumerate( eval_data_loader ):
            images = to_var( images )
            image_concepts = to_var( image_concepts )
            all_hyp, all_scores = Caption_Generator.translate_batch( images, image_concepts )

            # Build caption based on Vocabulary and the '<end>' token
            for image_idx in range( len( all_hyp ) ):

                all_sentence = []
                for num_i in range( args.n_best ):
                    sampled_ids = all_hyp[ image_idx ][ num_i ]
                    sampled_caption = []

                    for word_id in sampled_ids:

                        word = vocab.idx2word[ word_id ]
                        if word == '<end>':
                            break
                        else:
                            sampled_caption.append( word )

                    sentence = ' '.join( sampled_caption )
                    all_sentence.append( sentence )
                    
                best_sentence = all_sentence[0]
                temp = { 'image_id': int( image_ids[ image_idx ] ), 'caption': best_sentence }
                results.append( temp )

            # Disp evaluation process
            if (i+1) % (1000/args.eval_batch_size) == 0:
                print '[%d/%d]'%( (i+1),len( eval_data_loader ) )

        print '------------------------Caption Generated-------------------------------------'

        # Evaluate the results based on the COCO API
        resFile = os.path.join( args.result_path, args.basic_model, 'Caption-%d.json'%( epoch ) )
        json.dump( results, open( resFile , 'w' ) )

        annFile = args.caption_test_path
        coco = COCO( annFile )
        cocoRes = coco.loadRes( resFile )

        cocoEval = COCOEvalCap( coco, cocoRes )
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()

        print '-----------Evaluation performance on MS-COCO dataset----------'
        
        if args.use_MIA:
            save_file = args.save_score_file + '-' + args.basic_model + "-MIA"
        else:
            save_file = args.save_score_file + '-' + args.basic_model 
            
        f = open( save_file, 'a' )
        
        f.write('\n The evaluation scores about epoch %d are: \n'%( epoch ))
        for metric, score in cocoEval.eval.items():
            f.write('\n%s: %.4f\n'%( metric, score ))
            print '%s: %.4f'%( metric, score )
            
        f.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='self', help='To make it runnable in jupyter')
    parser.add_argument( '--save_dir_path', type=str, default='./ckpt',
                         help='path for saving trained models')
    parser.add_argument( '--basic_model', type=str, default='VisualAttention',
                         help='the selected basic model, [VisualAttention, ConceptAttention, VisualCondition, ConceptCondition]')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--result_path', type=str, default='./results',
                        help='path for saving generated captions')
    parser.add_argument('--save_score_file', type=str, default='./evaluation_scores',
                        help='path for saving the evaluation performance')
    parser.add_argument('--image_dir', type=str, default='./data/images/resized',
                        help='directory for resized training images')
    parser.add_argument('--caption_val_path', type=str,
                        default='./data/annotations/karpathy_split_val.json',
                        help='path for validation annotation json file')
    parser.add_argument('--caption_test_path', type=str,
                        default='./data/annotations/karpathy_split_test.json',
                        help='path for test annotation json file')
    parser.add_argument('--concept_path', type=str,
                        default='./data/image_concepts.json',
                        help='path for image concepts json file')

    # ---------------------------Hyper Parameter Setup------------------------------------
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size')
    parser.add_argument('--max_caption_length', type=int, default=20, help='The maximum length of generated captions.')
    parser.add_argument('--n_best', type=int, default=1, help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument( '--use_MIA', type=bool, default=False )
    parser.add_argument( '--iteration_times', type=int, default=2, help='the iteration times in mutual iterative attention' )
    parser.add_argument('--embed_size', type=int, default=512,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--eval_batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    print '------------------------Model and Testing Details--------------------------'
    print(args)

    # Start training
    main(args)

