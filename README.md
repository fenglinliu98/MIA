# MIA （NeurIPS 2019）
Implementation of "Aligning Visual Regions and Textual Concepts for Semantic-Grounded Image Representations" by Fenglin Liu, Yuanxin Liu, Xuancheng Ren, Xiaodong He, and Xu Sun. The paper can be found at [[arxiv]](https://arxiv.org/abs/1905.06139),  [[pdf]](https://papers.nips.cc/paper/8909-aligning-visual-regions-and-textual-concepts-for-semantic-grounded-image-representations.pdf).

## Semantic-Grounded Image Representations (Based on the Bottom-up features)
###  Coming Soon!
*  [2014 Train/Val Semantic-Grounded Image Representations]()

*  [2014 Testing Semantic-Grounded Image Representations]()

## Usage

### Requirements
This code is written in Python2.7 and requires PyTorch >= 0.4.1

You need to download pre-trained Resnet152 model from [torchvision](https://github.com/pytorch/vision) for both training and evaluation. 

You may take a look at https://github.com/s-gupta/visual-concepts to find how to get the topic words of an image.

### Training
Now we can train the baseline w/ MIA models with 

* **Visual Attention**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model VisualAttention
```

* **Concept Attention**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model ConceptAttention
```

* **Visual Condition**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model VisualCondition
```

* **Concept Attention**
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model ConceptCondition
```

* **Visual Regional Attention**
```
CUDA_VISIBLE_DEVICES=0 python train.py --basic_model VisualRegionalAttention
```

### Testing
We can test the trained model with 

```
CUDA_VISIBLE_DEVICES=0 python test.py --basic_model basic_model_name
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


