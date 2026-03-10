<p align="center">
<h1 align="center"><strong>HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control</strong></h1>
  <p align="center">
    <a href="https://bwrooney82.github.io/" target="_blank">Jinrui Han</a><sup>* 1,2</sup>,
    <a href="https://scholar.google.com/citations?user=iDv038AAAAAJ&hl=zh-CN" target="_blank">Dewei Wang</a><sup>* 1,3</sup>,
    <a href="https://github.com/chenyunzhanggit" target="_blank">Chenyun Zhang</a><sup>1</sup>,
    <a href="https://xinzheliu.github.io/" target="_blank">Xinzhe Liu</a><sup>1,4</sup>,
    <a href="http://luoping.me/" target="_blank">Ping Luo</a><sup>5</sup>,
    <a href="https://baichenjia.github.io/" target="_blank">Chenjia Bai</a><sup>&dagger;1</sup>,
    <a href="https://scholar.google.com.hk/citations?user=ahUibskAAAAJ" target="_blank">Xuelong Li</a><sup>&dagger;1</sup>
    <br>
    * Equal Contribution  &dagger; Corresponding Author
    <br>
    <sup>1</sup>Institute of Artificial Intelligence (TeleAI), China Telecom  
    <sup>2</sup>Shanghai Jiao Tong University  
    <sup>3</sup>University of Science and Technology of China  
    <sup>4</sup>ShanghaiTech University  
    <sup>5</sup>The University of Hong Kong
  </p>
</p>

<div id="top" align="center">

[![mjlab](https://img.shields.io/badge/mjlab-1.0.0-silver)](https://mujocolab.github.io/mjlab/main/index.html)
[![arXiv](https://img.shields.io/badge/Arxiv-2602.03205-A42C25?style=flat&logo=arXiv&logoColor=A42C25)](https://arxiv.org/abs/2602.03205)
[![PDF](https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow)](https://arxiv.org/pdf/2602.03205.pdf)
[![project](https://img.shields.io/badge/%F0%9F%9B%B9_Project-Page-lightblue)](https://husky-humanoid.github.io/)
[![Twitter](https://img.shields.io/badge/X-Husky-1DA1F2?logo=x&logoColor=white)](https://x.com/jinrui82)</div>

## Overview

[![method](media/method.jpg "method")]()


This repository contains the official implementation of our paper: [HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control](https://husky-humanoid.github.io/). In this work, we propose a learning-based whole-body control framework that empowers humanoid robots to perform dynamic skateboarding.

This repository contains:
- The mjlab training framework ([`src/mjlab_husky`](src/mjlab_husky))
- Customized RL implementations ([`rsl_rl/`](rsl_rl/))
- Motion data for AMP and trajectory planning ([`dataset/`](dataset/))
- Lightweight MuJoCo evaluation scripts ([`test_scene/`](test_scene/))
- Tested checkpoints ([`ckpts/`](ckpts/))

## Install
This code has been tested on Ubuntu 22.04 with CUDA 13.0.
To install this repository, please follow these steps:

1. **Install the [`uv`](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) package manager**  (if you don't have it yet):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:

   ```bash
   git clone https://github.com/TeleHuman/humanoid_skateboarding.git
   cd humanoid_skateboarding
   ```

3. **Sync dependencies**:

   ```bash
   uv sync
   uv pip install -e .
   ```

## Training Example

```bash
uv run train Mjlab-Skater-Flat-Unitree-G1 --env.scene.num-envs 4096
```

## Play Examples

```bash
uv run play Mjlab-Skater-Flat-Unitree-G1 --checkpoint_file your-ckpt-path
```

We also provide a lite MuJoCo simulation script for evaluation:

```bash
bash test_scene/sim.sh your-onnx-path
```

The [`test_scene/mjlab_scene.xml`](test_scene/mjlab_scene.xml) file is automatically generated from the mjlab [`scene_cfg`](src/mjlab_husky/asset_zoo/robots/skateboard/g1_skater_constants.py). In the simulation, you can control the skateboard using the keyboard arrow keys. Visualization examples are shown below, rendered from [`test.pt`](ckpts/test.pt) and [`test.onnx`](ckpts/test.onnx):

| <div align="center">  Viser </div>                                                                                                                                           | <div align="center"> MuJoCo </div>                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="media/viser.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="media/mjc.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>

## Citation

If you find our work helpful, please consider citing us:

```bibtex
@article{han2026husky,
    title={HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control},
    author={Jinrui Han and Dewei Wang and Chenyun Zhang and Xinzhe Liu and Ping Luo and Chenjia Bai and Xuelong Li},
    journal={arXiv preprint arXiv:2602.03205},
    year={2026}
  }
```

## License

This codebase is under [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en). You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products.

## Acknowledgements
- [mjlab](https://github.com/mujocolab/mjlab): Our training framework is based on `mjlab` by MuJoCo Lab.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl): The reinforcement learning algorithm is built upon the `rsl_rl` library.
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): GPU-accelerated interface for rendering and physics simulation.
- [mujoco](https://github.com/google-deepmind/mujoco.git): High-fidelity rigid-body physics engine.
- [AMP](https://github.com/xbpeng/MimicKit): We build on Adversarial Motion Priors for pushing behaviors.
- [DHAL](https://github.com/UMich-CURLY/DHAL): We drew inspiration from the quadrupedal robot skateboarding project.

## Contact

For further collaborations or discussions, please feel free to reach out to:

- Jinrui Han: [jrhan82@sjtu.edu.cn](mailto:jrhan82@sjtu.edu.cn) or Wechat: [Bw_rooneY](https://bwrooney82.github.io/assets/img/wechat.png)
- Chenjia Bai (Corresponding Author): [baicj@chinatelecom.cn](mailto:baicj@chinatelecom.cn)
