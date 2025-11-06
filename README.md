<div align="center">
<h2 align="center">CAGE: Continuity-Aware edGE Network Unlocks Robust Floorplan Reconstruction</h2>
<h3 align="center">Neurips 2025</h3>
<!-- Yiyi Liu, Chunyang Liu, Bohan Wang, Weiqin Jiao,  -->

<!-- Bojian Wu, Lubin Fan, Yuwei Chen, Fashuai Li, Biao Xiong -->

<!-- <a href="https://theodorakontogianni.github.io/">Theodora Kontogianni</a>, <a href="https://igp.ethz.ch/personen/person-detail.html?persid=143986">Konrad Schindler</a>, <a href="https://francisengelmann.github.io/">Francis Engelmann</a> -->

<!-- Wuhan University of Technology  
University of Twente  
Hangzhou Institute for Advanced Study, University of Chinese Academy of Sciences  
The Advanced Laser Technology Laboratory of Anhui Province -->


<!-- ![teaser](./imgs/teaser.jpg) -->
<img src="./imgs/teaser.svg" width=100% height=100%>

</div>



[[Project Webpage](https://ywyue.github.io/RoomFormer/)]    [[Paper](https://arxiv.org/abs/2211.15658)]    [[Video](https://www.youtube.com/watch?v=yzYe4yVN1NU)]


<!-- <details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#method">Method</a>
    </li>
    <li>
      <a href="#preparation">Preparation</a>
    </li>
    <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#training">Training</a>
    </li>
    <li>
      <a href="#semantically-rich-floorplan">Semantically-rich Floorplan</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#acknowledgment">Acknowledgment</a>
    </li>
  </ol>
</details> -->

<!-- 
## Abstract

We address 2D floorplan reconstruction from 3D scans. Existing approaches typically employ heuristically designed multi-stage pipelines. Instead, we formulate floorplan reconstruction as a single-stage structured prediction task: find a variable-size set of polygons, which in turn are variable-length sequences of ordered vertices. To solve it we develop a novel Transformer architecture that generates polygons of multiple rooms in parallel, in a holistic manner without hand-crafted intermediate stages. The model features two-level queries for polygons and corners, and includes polygon matching to make the network end-to-end trainable. Our method achieves a new state-of-the-art for two challenging datasets, Structured3D and SceneCAD, along with significantly faster inference than previous methods. Moreover, it can readily be extended to predict additional information, i.e., semantic room types and architectural elements like doors and windows.


## Method
 ![space-1.jpg](./imgs/model.gif) 

**Illustration of the RoomFormer model**. Given a top-down-view density map of the input point cloud, (a) the feature backbone extracts multi-scale features, adds positional encodings, and flattens them before passing them into the (b) Transformer encoder. (c) The Transformer decoder takes as input our two-level queries, one level for the room polygons (up to M) and one level for their corners (up to N per room polygon). A feed-forward network (FFN) predicts a class c for each query to accommodate for varying numbers of rooms and corners. During training, the polygon matching guarantees optimal assignment between predicted and groundtruth polygons. -->


## Preparation
### Environment
* The code has been tested on Linux with python 3.8, torch 1.9.0, and cuda 11.1.
* We recommend an installation through conda:
  * Create an environment:
  ```shell
  conda create -n cage python=3.8
  conda activate cage
  ```
  * Install pytorch and other required packages:
  ```shell
  # adjust the cuda version accordingly
  pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  pip install -r requirements.txt
  ```
  * Compile the deformable-attention modules (from [deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)) and the differentiable rasterization module (from [BoundaryFormer](https://github.com/mlpc-ucsd/BoundaryFormer)):
  ```shell
  cd models/ops
  sh make.sh

  # unit test for deformable-attention modules (should see all checking is True)
  # python test.py

  cd ../../diff_ras
  python setup.py build develop
  ```


### Data

We follow the official data format provided by [RoomFormer](data_preprocess) and directly use their processed data. All preprocessing steps are conducted as described in data_preprocess
.

### Backbone

CAGE support two backbone architectures: **ResNet-50** and **[Swin Transformer](https://github.com/microsoft/Swin-Transformer?tab=readme-ov-file#main-results-on-imagenet-with-pretrained-models)**.
Please set the backbone type and corresponding hyperparameters accordingly.

* **ResNet-50**  
  Set ```backbone=resnet50``` in the corresponding script under the ```tools/``` directory or in main.py.  
  No additional setup is required.

* **Swin Transformer**  
  * Set ```backbone=swinv2_L_192_22k``` in the corresponding script under the ```tools/``` directory or in main.py.  
  * Place the pretrained [Swin Transformer weight](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth) in the ``` pretrained/ ```folder.

### Checkpoints

Please download and extract the checkpoints of our model from [this link](https://drive.google.com/drive/folders/1jajjRamJ7SVgCWB-Tihp-ToqPsv0GmE7?usp=sharing).


## Evaluation

#### Structured3D
We use the same evaluation scripts with [MonteFloor](https://openaccess.thecvf.com/content/ICCV2021/papers/Stekovic_MonteFloor_Extending_MCTS_for_Reconstructing_Accurate_Large-Scale_Floor_Plans_ICCV_2021_paper.pdf). Please first download the ground truth data used by [MonteFloor](https://openaccess.thecvf.com/content/ICCV2021/papers/Stekovic_MonteFloor_Extending_MCTS_for_Reconstructing_Accurate_Large-Scale_Floor_Plans_ICCV_2021_paper.pdf) and [HEAT](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_HEAT_Holistic_Edge_Attention_Transformer_for_Structured_Reconstruction_CVPR_2022_paper.pdf) with [this link](https://drive.google.com/file/d/1XpKm3vjvw4lOw32pX81w0U0YL_PBuzez/view?usp=sharing) (required by the evaluation code) and extract it as ```./s3d_floorplan_eval/montefloor_data```. Then run following command to evaluate the model on Structured3D test set:
```shell
./tools/eval_stru3d.sh
```

#### SceneCAD
We adapt the evaluation scripts from MonteFloor to evaluate SceneCAD:
```shell
./tools/eval_scenecad.sh
```

## Training
The command for training RoomFormer on Structured3D is as follows:
```shell
./tools/train_stru3d.sh
```
Similarly, to train RoomFormer on SceneCAD, run the following command:
```shell
./tools/train_scenecad.sh
```

<!-- 
## Semantically-rich Floorplan
RoomFormer can be easily extended to predict room types, doors and windows. We provide the implementation and model for SD-TQ (The variant with minimal changes to our original architecture). To evaluate or train on the semantically-rich floorplans of Structured3D, run the following commands:
```shell
### Evaluation:
./tools/eval_stru3d_sem_rich.sh
### Train:
./tools/train_stru3d_sem_rich.sh
``` -->

## Citation
If you find CAGE useful in your research, please cite our paper:
```
@inproceedings{liu2025cage,
  title     = {CAGE: Continuity-Aware edGE Network Unlocks Robust Floorplan Reconstruction},
  author    = {Liu, Yiyi and Liu, Chunyang and Jiao, Weiqin and Wu, Bojian and Li, Fashuai and Xiong, Biao},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

## Acknowledgment

We thank the authors of FRI-Net, PolyRoom, RoomFormer, HEAT and MonteFloor for providing results on Structured3D for better comparison. We also thank for the following excellent open source projects:
* [FRI-Net](https://github.com/Daisy-1227/FRI-Net)
* [PolyRoom](https://github.com/3dv-casia/PolyRoom)
* [RoomFormer](https://github.com/ywyue/RoomFormer)
* [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
* [DETR](https://github.com/facebookresearch/detr)
* [DN-DETR](https://github.com/IDEA-Research/DN-DETR)
* [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
* [Detectron2](https://github.com/facebookresearch/detectron2)
* [HEAT](https://github.com/woodfrog/heat)
* [BoundaryFormer](https://github.com/mlpc-ucsd/BoundaryFormer)


