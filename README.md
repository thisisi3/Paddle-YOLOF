## Introduction

In this project we reproduce YOLOF, a one stage object detector proposed in paper [You Only Look One-level Feature](https://arxiv.org/abs/2103.09460), based on [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection). 

YOLOF uses only one feature level to achieve competitive accuracy as those who use multiple feature levels (FPN). It is largely due to the following two novel designs:

**Dilated Encoder:** use dilated convolution and residual blocks to enlarge receptive fields while still keep multiple receptive fields.

**Uniform Assigner:** assign k nearest anchors to each GT to ensure GTs have balanced positive samples.

One benefit of using only one feature level is reduce of FLOPs and increase of speed. One data shows YOLOF achieves the same AP on COCO as multi-level-feature RetinaNet while has 57% less FLOPs and 2.5x speed up. We strongly recommend this paper, please visit [here](https://arxiv.org/abs/2103.09460) to check it out.

There are two official implementations, [one](https://github.com/chensnathan/YOLOF) is based on Detectron2 and the [other](https://github.com/Megvii-BaseDetection/cvpods) is based on cvpods. MMDetection has also implemented YOLOF and included it in their model list. Here we follow both official's Detectron2 version and MMDetection's version. 

Note that our implementation is based on PaddleDetection which is built on the deep learning platform of [Paddle](https://github.com/PaddlePaddle/Paddle). 



## Results

 

| source                                                       | backbone | AP   | epochs                                                       | config                                                       | model                                                        | train-log                                                    | dataset  |
| ------------------------------------------------------------ | -------- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| [official](https://github.com/chensnathan/YOLOF)             | R-50-C5  | 37.7 | 12.3([detail](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolof/yolof_r50_c5_8x8_iter-1x_coco.py)) | [config](https://github.com/chensnathan/YOLOF/blob/master/configs/yolof_R_50_C5_1x.yaml) | [model](https://pan.baidu.com/share/init?surl=BSOncRYq6HeCQ8q2hrWowA)[qr6o] | NA                                                           | coco2017 |
| [mmdet](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolof) | R-50-C5  | 37.5 | 12                                                           | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolof/yolof_r50_c5_8x8_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth) | [log](https://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427.log.json) | coco2017 |
| this                                                         | R-50-C5  | 37.5 | 12                                                           | [config](https://github.com/thisisi3/Paddle-YOLOF/blob/main/configs/yolof_r50_c5_1x_coco_8x4GPU.yml) | [model](https://pan.baidu.com/s/1LiDK0V40BwyucFZDoJ3crA)[3z7q] | [log](https://github.com/thisisi3/Paddle-YOLOF/blob/main/train-log-37.4.txt) | coco2017 |
| this_re-train                                                | R-50-C5  | 37.4 | 12                                                           | [config](https://github.com/thisisi3/Paddle-YOLOF/blob/main/configs/yolof_r50_c5_1x_coco_8x4GPU.yml) | [model](https://pan.baidu.com/s/1d0RXl2GVoQ77kg_7zzePfQ)[6faq] | [log](https://github.com/thisisi3/Paddle-YOLOF/blob/main/train-log-37.5.txt) | coco2017 |

We train and test our implementation on coco 2017 dataset. The models we provide here are trained on Baidu AIStudio platform. They are trained on 4 V100 GPUs with 8 images per GPU. Data in first two rows above table is directly taken from their official github repos. According to MMDetection's [comment](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolof), both mmdet and official's version have 0.3 variation of AP. So we re-trained the same config and got 37.4 AP. Thank the team of MMDetection for providing such important information. 

Please check out the config for more information on the model.

Please check out the train-log for more information on the loss during training.

## Code

The implementation is based on PaddleDetection v2.3, the directory `PaddleDetection/` basically contains the whole code base of PaddleDetection. All code related to YOLOF is located at `PaddleDetection/ppdet/yolof`.



## Usage

**Requirements:**

- python 3.7+
- Paddle v2.2: follow [this](https://www.paddlepaddle.org.cn/install/quick) to install

**Clone this repo and install:**

```shell
git clone https://github.com/thisisi3/Paddle-YOLOF.git
pip install -e Paddle-YOLOF/PaddleDetection -v
```

Follow [this](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/INSTALL.md) for detailed steps on installation of PaddleDetection and follow [this](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/GETTING_STARTED.md) to learn how to use PaddleDetection.

**Data preparation:**

```shell
cd Paddle-YOLOF

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

mkdir dataset
mkdir dataset/coco

unzip annotations_trainval2017.zip -d dataset/coco
unzip train2017.zip -d dataset/coco
unzip val2017.zip -d dataset/coco
```

You can also go to [aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/7122) to download coco 2017 if official download is slow.

**Download pretrained backbone:**

YOLOF uses caffe-style ResNet, it corresponds to variant-a in PaddleDetection. But PaddleDetection does not come with pretrained weight of such variant, so we manually converted the weight and you can download it [here](https://pan.baidu.com/s/1BOtYycALwz55QJsmbtsPoQ)[rpsb]. After you download the weight please put it under directory `pretrain/`.

**Train YOLOF on a single GPU:**

```shell
python PaddleDetection/tools/train.py -c configs/yolof_r50_c5_1x_coco_8x4GPU.yml --eval
```

**Train YOLOF on 4 GPUs:**

```shell
python -m paddle.distributed.launch --gpus 0,1,2,3 PaddleDetection/tools/train.py -c configs/yolof_r50_c5_1x_coco_8x4GPU.yml --eval
```

If you do not want to evaluate AP during training, simply remove the `--eval` option.

**Eval AP of YOLOF:**

```shell
python PaddleDetection/tools/eval.py -c configs/yolof_r50_c5_1x_coco_8x4GPU.yml -o weights=path_to_model_final.pdparams
```

**Quick demo:**

Thanks to PaddleDetection, we can use the inference script `PaddleDetection/tools/infer.py` they provide to visualize detection results of  YOLOF, by running the following code:

```shell
python PaddleDetection/tools/infer.py -c configs/yolof_r50_c5_1x_coco_8x4GPU.yml -o weights=path_to_model_final.pdparams --infer_img demo/000000185250.jpg --output_dir demo/out/ --draw_threshold 0.5
```

The test image:

![](https://github.com/thisisi3/Paddle-YOLOF/blob/main/demo/000000185250.jpg?raw=true)

After adding bboxes:

![](https://github.com/thisisi3/Paddle-YOLOF/blob/main/demo/out/000000185250.jpg?raw=true)

Both images can be found at `demo/`.

## Acknowledgement

We would like to thank Baidu AIStudio for providing good quality and good amount of GPU power. 

Also thank the following amazing open-source projects:

- [YOLOF-Detection2](https://github.com/chensnathan/YOLOF)

- [YOLOF-cvpods](https://github.com/Megvii-BaseDetection/cvpods)

- [YOLOF-MMDetection](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolof)

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

