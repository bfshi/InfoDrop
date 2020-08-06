# A Closer Look at Few-shot Classification

This repo is based on the [official code](https://github.com/wyharveychen/CloserLookFewShot) for the paper [A Closer Look at Few-shot Classification](https://openreview.net/pdf?id=HkxLXnAcFQ) (ICLR 2019). 

## Enviroment
 - Python3
 - [Pytorch](http://pytorch.org/)
 - json

## Getting started
### CUB
* Change directory to `./filelists/CUB`
* run `source ./download_CUB.sh`

### mini-ImageNet
* Change directory to `./filelists/miniImagenet`
* run `source ./download_miniImagenet.sh` 

(WARNING: This would download the 155G ImageNet dataset. You can comment out correponded line 5-6 in `download_miniImagenet.sh` if you already have one.) 

### mini-ImageNet->CUB (cross)
* Finish preparation for CUB and mini-ImageNet and you are done!

### Omniglot
* Change directory to `./filelists/omniglot`
* run `source ./download_omniglot.sh` 

### Omniglot->EMNIST (cross_char)
* Finish preparation for omniglot first
* Change directory to `./filelists/emnist`
* run `source ./download_emnist.sh`  

### Self-defined setting
* Require three data split json file: 'base.json', 'val.json', 'novel.json' for each dataset  
* The format should follow   
{"label_names": ["class0","class1",...], "image_names": ["filepath1","filepath2",...],"image_labels":[l1,l2,l3,...]}  
See test.json for reference
* Put these file in the same folder and change data_dir['DATASETNAME'] in configs.py to the folder path  

## Train

Before training, you can tweak the hyperparameters for InfoDrop in `backbone.py` (L111-112, L125-130)

For example, to train protonet (w/ InfoDrop) on CUB, first run:

`python ./train.py --method protonet --stop_epoch 600`  

then remove InfoDrop (change `finetune_wo_infodrop` to `True` in `backbone.py`, L112) and finetune:

`python ./train.py --method protonet --stop_epoch 612 --resume_acc [] `

Please refer to `io_utils.py` for details of all the arguments.

## Save features
Save the extracted feature before the classifaction layer to increase test speed. This is not applicable to MAML, but are required for other methods.
Run
```python ./save_features.py --method protonet --best_acc [] ```

## Test
Run
`python ./test.py --method protonet --best_acc []`

## Results
* The test results will be recorded in `./record/results.txt`
* For all the pre-computed results, please see `./record/few_shot_exp_figures.xlsx`. This will be helpful for including your own results for a fair comparison.

## References
Our testbed builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:

* Framework, Backbone, Method: Matching Network
https://github.com/facebookresearch/low-shot-shrink-hallucinate 
* Omniglot dataset, Method: Prototypical Network
https://github.com/jakesnell/prototypical-networks
* Method: Relational Network
https://github.com/floodsung/LearningToCompare_FSL
* Method: MAML
https://github.com/cbfinn/maml  
https://github.com/dragen1860/MAML-Pytorch  
https://github.com/katerakelly/pytorch-maml
