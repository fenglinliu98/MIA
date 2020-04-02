# MIA （NeurIPS 2019）
Implementation of "Aligning Visual Regions and Textual Concepts for Semantic-Grounded Image Representations" by Fenglin Liu, Yuanxin Liu, Xuancheng Ren, Xiaodong He, and Xu Sun. The paper can be found at [[arxiv]](https://arxiv.org/abs/1905.06139),  [[pdf]](https://papers.nips.cc/paper/8909-aligning-visual-regions-and-textual-concepts-for-semantic-grounded-image-representations.pdf).

![Image text](https://github.com/fenglinliu98/MIA/blob/master/model.png)

## Semantic-Grounded Image Representations (Based on the  [Bottom-up features](https://github.com/peteanderson80/bottom-up-attention))
###  Coming Soon!
*  [2014 Train/Val Semantic-Grounded Image Representations](https://github.com/fenglinliu98/MIA)


## Usage

### Requirements
This code is written in Python2.7 and requires PyTorch >= 0.4.1

 [Textual Concepts (Google Drive)](https://drive.google.com/open?id=1jpSZbLXD1Ev3OC2t_NFFvxYo40UcnV7Q)
 
You may take a look at https://github.com/s-gupta/visual-concepts to find how to get the textual concepts of an image by yourself.

### Dataset Preparation
#### Download MSCOCO images and preprocess them
Download the mscoco images from  [link](http://mscoco.org/dataset/#download). We need 2014 training images and 2014 val. images. You should put the train2014/ and val2014/ in the ./data/images/ directory.

We also provide a download bash script to download the mscoco images:
```
bash ./data/download_mscoco_images.sh
```

#### Download MSCOCO captions and preprocess them
You may download the mscoco captions from the [official website](http://mscoco.org/dataset/#download).

Note: We also provide a [link](https://github.com/fenglinliu98/MIA) to download our preprocessed mscoco captions.

### Start Training
Now we can train the baseline models and the baseline w/ MIA models with: 

#### Visual Attention
* **Baseline**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model VisualAttention --MIA False
```
* **Baseline w/ MIA**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model VisualAttention --MIA True
```

#### Concept Attention
* **Baseline**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model ConceptAttention --MIA False
```
* **Baseline w/ MIA**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model ConceptAttention --MIA True
```

#### Visual Condition
* **Baseline**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model VisualCondition --MIA False
```
* **Baseline w/ MIA**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model VisualCondition --MIA True
```

#### Concept Attention
* **Baseline**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model ConceptCondition --MIA False
```
* **Baseline w/ MIA**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model ConceptCondition --MIA True
```

#### Visual Regional Attention
* **Baseline**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model VisualRegionalAttention --MIA False
```
* **Baseline w/ MIA**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model VisualRegionalAttention --MIA True
```

### Testing
We can test the trained model with 
* **Baseline**
```
CUDA_VISIBLE_DEVICES=0 python test.py  --basic_model basic_model_name --MIA False
```
* **Baseline w/ MIA**
```
CUDA_VISIBLE_DEVICES=0 python test.py  --basic_model basic_model_name --MIA True
```

## Reference
If you use this code as part of any published research, please acknowledge the following paper
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


