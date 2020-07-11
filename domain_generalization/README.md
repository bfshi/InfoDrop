# Domain Generalization
This repo is based on the [official code](https://github.com/fmcarlucci/JigenDG) from the CVPR19 oral paper "[Domain Generalization by Solving Jigsaw Puzzles](https://arxiv.org/pdf/1903.06864.pdf)".

## SETUP 

PACS dataset can be downloaded from [here](https://drive.google.com/file/d/0B6x7gtvErXgfbF9CSk53UkRxVzg/view?usp=sharing). Once you have download the data, you must update the files in data/txt_list to match the actual location of your files.
For example, if you saved your data into /home/user/data/images/ you have to change these lines:

```
/home/fmc/data/PACS/kfold/art_painting/dog/pic_001.jpg 0
/home/fmc/data/PACS/kfold/art_painting/dog/pic_002.jpg 0
/home/fmc/data/PACS/kfold/art_painting/dog/pic_003.jpg 0
```

into:

```
/home/user/data/images/PACS/kfold/art_painting/dog/pic_001.jpg 0
/home/user/data/images/PACS/kfold/art_painting/dog/pic_002.jpg 0
/home/user/data/images/PACS/kfold/art_painting/dog/pic_003.jpg 0
```

A quick way is to use sed:
`for i in *.txt; do sed -i "s@/home/fmc/data/@/home/user/data/images/@g" $i; done`

Pytorch models will automatically download if needed. You can download the caffe model we used for AlexNet from here https://drive.google.com/file/d/1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-/view?usp=sharing

Once downloaded, move it into models/pretrained/alexnet_caffe.pth.tar




## Running experiments

Commands for running experiments with different source/target domains are listed in `scripts`. 

You can change the hyperparameters for InfoDrop in `models/resnet.py`. (L33-34 and L47-52)

For example, train a model (w/ InfoDrop) with sketch as source domain and photo as target domain, first run:

```
python train_jigsaw.py --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source sketch --target photo --jig_weight 0.0 --bias_whole_image 0.9 --image_size 222 --classify_only_sane --epochs 100
```

then remove InfoDrop (turn the `finetune_wo_infodrop` on in L34 of `models/resnet.py`), and finetune:

```
python train_jigsaw.py --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source sketch --target photo --jig_weight 0.0 --bias_whole_image 0.9 --image_size 222 --classify_only_sane --epochs 20 --checkpoint [PATH TO CHECKPOINT]
```

