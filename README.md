# Monte Carlo Marginalization: A Differentiable Method to Learn High-Dimensional Distributions

[![Paper](https://img.shields.io/badge/IEEE-TNNLS-blue)](https://ieeexplore.ieee.org/abstract/document/11494096)

Official implementation of the paper: **"Monte Carlo Marginalization: A Differentiable Method to Learn High-Dimensional Distributions"**, published in *IEEE Transactions on Neural Networks and Learning Systems (TNNLS)*, 2026.

---

![MCMarg Training Process](./updateGMM.gif)
*Visualization of the distribution learning process using MCMarg.*

## Introduction

**MCMarg** is a novel differentiable framework designed to learn complex, high-dimensional probability distributions. By integrating Monte Carlo techniques into a gradient-based optimization pipeline, our method addresses the "curse of dimensionality" commonly encountered in generative modeling, filtering, and density estimation.

## Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/MCMarg.git](https://github.com/your-username/MCMarg.git)
cd MCMarg

# Install dependencies (Example)
pip install torch torchvision numpy
