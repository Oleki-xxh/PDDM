# PDDM: Pseudo Depth Diffusion Model for RGB-PD Semantic Segmentation Based in Complex Indoor Scenes (AAAI 2025)

Implementation of PDDM in [PDDM: Pseudo Depth Diffusion Model for RGB-PD Semantic Segmentation Based in Complex Indoor Scenes](https://ojs.aaai.org/index.php/AAAI/article/view/32970). 


* [0. Table of Contents](#0-table-of-contents)

* [1. Run the Code](#1-run-the-code)

* [2. Acknowledgement](#2-Acknowledgement)

* [3. Citation](#3-citation)



## 1. Run the Code

1. Installation

```shell
conda create -n pddm python=3.10

conda activate pddm

pip install -r requirements.txt

pip install -e .

conda install pytorch=2.0.1 cudatoolkit=11.7 -c pytorch -c conda-forge

git clone https://github.com/facebookresearch/detectron2.git

python -m pip install -e detectron2
```

2. Datasets

Download the dataset [NYUv2](https://drive.google.com/file/d/10d7oB5q4k6eLHVcb_2DXzLJwhlQqIqZL/view?usp=drive_link) and place it in the folder PDDM/EMSANet/datasets.

3. Train

```shell
CUDA_VISIBLE_DEVICES=0 python ./train_net.py --config-file configs/semantic/PDDM_NYUv2.py --num-gpus 1 --amp --ref 1
```

It needs 24G to train with 2 batch size.

4. Evaluation

```shell
CUDA_VISIBLE_DEVICES=0 python ./train_net.py --config-file configs/semantic/PDDM_NYUv2.py --num-gpus 1 --amp --ref 1 --init-from /path/to/checkpoint --eval-only
```

We release the trained [checkpoint](https://drive.google.com/file/d/1D0b5qDzy_-od0BSecsG1EOdFbQn85uG-/view?usp=drive_link) on NYUv2.

## 2. Acknowledgement


Code is largely based on [ODISE](https://github.com/NVlabs/ODISE), [EMSANet](https://github.com/TUI-NICR/EMSANet), [Detectron2](https://github.com/facebookresearch/detectron2), [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [OpenCLIP](https://github.com/mlfoundations/open_clip) and [GLIDE](https://github.com/openai/glide-text2im).

Thank you for the great open-source projects!


## 3. Citation

If you find this work helpful in your research, please consider citing:

```
@inproceedings{xu2025pddm,
  title={PDDM: Pseudo Depth Diffusion Model for RGB-PD Semantic Segmentation Based in Complex Indoor Scenes},
  author={Xu, Xinhua and Liu, Hong and Wu, Jianbing and Liu, Jinfu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={9},
  pages={8969--8977},
  year={2025}
}
```

