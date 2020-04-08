# MIA （NeurIPS 2019）
Implementation of "Aligning Visual Regions and Textual Concepts for Semantic-Grounded Image Representations" by Fenglin Liu, Yuanxin Liu, Xuancheng Ren, Xiaodong He, and Xu Sun. The paper can be found at [[arxiv]](https://arxiv.org/abs/1905.06139),  [[pdf]](https://papers.nips.cc/paper/8909-aligning-visual-regions-and-textual-concepts-for-semantic-grounded-image-representations.pdf).

![Image text](https://github.com/fenglinliu98/MIA/blob/master/model.png)

## Semantic-Grounded Image Representations (Based on the  [Bottom-up features](https://github.com/peteanderson80/bottom-up-attention))
###  Coming Soon!
*  [2014 Train/Val Semantic-Grounded Image Representations](https://github.com/fenglinliu98/MIA)

## [Textual Concepts (Google Drive)](https://drive.google.com/open?id=1jpSZbLXD1Ev3OC2t_NFFvxYo40UcnV7Q)

## Usage

### Requirements
This code is written in Python2.7 and requires PyTorch >= 0.4.1
 
You may take a look at https://github.com/s-gupta/visual-concepts to find how to get the textual concepts of an image by yourself.

### Dataset Preparation
#### Download MSCOCO images and preprocess them
* **Download**

Download the mscoco images from  [link](http://mscoco.org/dataset/#download). We need 2014 training images and 2014 val. images. You should put the train2014/ and val2014/ in the ./data/images/ directory.

Note: We also provide a download bash script to download the mscoco images:
```
cd data/images/original && bash download_mscoco_images.sh
```

* **Preprocess**

Now we may need to run resize.py to resize all the images (in both train and val folder) into 256 x 256. You may specify different locations inside resize.py
```
python resize_images.py
```

#### Download MSCOCO captions and preprocess them
* **Download**

You may download the mscoco captions from the [official website](http://mscoco.org/dataset/#download) or use the download bash script provided by us.
```
cd data && bash download_mscoco_captions.sh
```

* **Preprocess**

Afterwards, we should create the Karpathy split for training, validation and test.
```
python KarpathySplit.py
```

Then we can build the vocabulary by running
(Note: You should download the [nltk_data](https://drive.google.com/open?id=1W95OMsG71cRMdMetIEGZg-fuPrSk6iVw) to build the vocabulary.)
```
unzip nltk_data.zip && python build_vocab.py
```

#### Download image concepts
Download the [Textual Concepts (Google Drive)](https://drive.google.com/open?id=1jpSZbLXD1Ev3OC2t_NFFvxYo40UcnV7Q) and put it in the ./data/ directory.
```
mv image_concepts.json ./data
```

### Start Training
Now we can train the baseline models and the baseline w/ MIA models with: 

#### Visual Attention
* **Baseline**
```
CUDA_VISIBLE_DEVICES=0,1 python Train.py --basic_model=VisualAttention 
```
* **Baseline w/ MIA**
```
CUDA_VISIBLE_DEVICES=0,1 python Train.py --basic_model=VisualAttention --use_MIA=True --iteration_times=2
```

#### Concept Attention
* **Baseline**
```
CUDA_VISIBLE_DEVICES=0,1 python Train.py --basic_model=ConceptAttention
```
* **Baseline w/ MIA**
```
CUDA_VISIBLE_DEVICES=0,1 python Train.py --basic_model=ConceptAttention --use_MIA=True --iteration_times=2
```

#### Visual Condition
* **Baseline**
```
CUDA_VISIBLE_DEVICES=0,1 python Train.py --basic_model=VisualCondition
```
* **Baseline w/ MIA**
```
CUDA_VISIBLE_DEVICES=0,1 python Train.py --basic_model=VisualCondition --use_MIA=True --iteration_times=2
```

#### Concept Attention
* **Baseline**
```
CUDA_VISIBLE_DEVICES=0,1 python Train.py --basic_model=ConceptCondition
```
* **Baseline w/ MIA**
```
CUDA_VISIBLE_DEVICES=0,1 python Train.py --basic_model=ConceptCondition --use_MIA=True --iteration_times=2
```

#### Visual Regional Attention
* **Baseline**
```
CUDA_VISIBLE_DEVICES=0,1 python Train.py --basic_model=VisualRegionalAttention
```
* **Baseline w/ MIA**
```
CUDA_VISIBLE_DEVICES=0,1 python Train.py --basic_model=VisualRegionalAttention --use_MIA=True --iteration_times=2
```

### Testing
We can test the trained model with 
* **Baseline**
```
CUDA_VISIBLE_DEVICES=0 python Test.py  --basic_model=basic_model_name
```
Note: basic_model_name = (VisualAttention, ConceptAttention, VisualCondition, ConceptCondition, VisualRegionalAttention)
* **Baseline w/ MIA**
```
CUDA_VISIBLE_DEVICES=0 python Test.py  --basic_model=basic_model_name --use_MIA=True --iteration_times=2
```

## Reference
If you use this code or our extracted image concepts as part of any published research, please acknowledge the following paper
```
@inproceedings{Liu2019MIA,
  author    = {Fenglin Liu and
               Yuanxin Liu and
               Xuancheng Ren and
               Xiaodong He and
               Xu Sun},
  title     = {Aligning Visual Regions and Textual Concepts for Semantic-Grounded
               Image Representations},
  booktitle = {NeurIPS},
  pages     = {6847--6857},
  year      = {2019}
}
```

## Acknowledgements

Thanks to [Torch](http://torch.ch/) team for providing Torch 0.4, [COCO](http://cocodataset.org/) team for providing dataset, [Tsung-Yi Lin](https://github.com/tylin/coco-caption) for providing evaluation codes for MS COCO caption generation, [Yufeng Ma](https://github.com/yufengm) for providing open source repositories and Torchvision [ResNet](https://github.com/pytorch/vision) implementation. 

### Note
If you have any questions about the code or our paper, please send an email to fenglinliu98@pku.edu.cn


