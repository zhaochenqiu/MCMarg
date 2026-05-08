# Monte Carlo Marginalization: A Differentiable Method to Learn High-Dimensional Distributions

[![Paper](https://img.shields.io/badge/IEEE-TNNLS-blue)](https://ieeexplore.ieee.org/abstract/document/11494096)

Implementation of the paper: **"Monte Carlo Marginalization: A Differentiable Method to Learn High-Dimensional Distributions"**, published in *IEEE Transactions on Neural Networks and Learning Systems (TNNLS)*, 2026.

---



## Introduction
**MCMarg** is a novel differentiable framework designed to learn complex, high-dimensional probability distributions. By integrating Monte Carlo techniques into a gradient-based optimization pipeline, our method addresses the "curse of dimensionality" commonly encountered in generative modeling, filtering, and density estimation.

<img src="./figure.png" width="100%">
> **Figure 1: Illustration of the proposed approach.** > We use a Gaussian Mixture Model $p(\mathbf{z};\boldsymbol{\theta})$ to approximate the target distribution $q(\mathbf{z})$, by minimizing the KL divergence between their marginal distributions $q_{\vec{u}}(\mathbf{z})$ and $p_{\vec{u}}(\mathbf{z})$ on different random unit vectors $\vec{u}_1, \vec{u}_2, \dots, \vec{u}_T$.




## Running the Code

### Default Execution
Run the model with default settings:
```
Bash
python MCMarg.py
```

### Specific Distribution Learning
To train the model on a specific dataset (e.g., the Moons dataset), use the --datapth argument to specify the path:
```
Bash
python MCMarg.py --datapth ./data/samples_moons.pt
```
## Citation

If you use this code or the MCMarg method in your research, please cite our paper:
```
@ARTICLE{11494096,
  author={Zhao, Chenqiu and Dong, Guanfang and Basu, Anup},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Monte Carlo Marginalization: A Differentiable Method to Learn High-Dimensional Distributions}, 
  year={2026},
  volume={},
  number={},
  pages={1-15},
  keywords={Filtering;Oscillators;Circuits and systems;Central Processing Unit;Filters;System-on-chip;Videos;Video equipment;Internet of Things;Communication systems;Distribution learning;image generation;Monte Carlo method},
  doi={10.1109/TNNLS.2026.3682991}
}
```

## Demo

<p align="center">
  <img src="./updateGMM.gif" width="100%">
</p>
