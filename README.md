# Component-Level PIML

This repository contains preliminary experiments from my Master's thesis exploring **Quantifying the Reduction in Carbon Footprint of Physics-Informed Machine Learning (PIML)** at the component level. The goal is to evaluate small-scale models with varying degrees of physics inductive bias to understand their impact on performance, generalization, and environmental footprint.

---

## Overview of Experiments

### `1_Interpolation`
This experiment explores the use of **physics-informed activation functions** to embed domain knowledge into MLP architectures. Four MLPs are compared: two with standard activation functions and two with physics-informed variants. Models are evaluated based on computational and data efficiency as well as model size.

### `2_Extrapolation`
Introduces a **Physics-Informed Neural Network (PINN)** where the loss function is augmented with terms from the governing differential equations. The PINN is compared to the MLPs from `1_Interpolation`, with emphasis on interpolation and extrapolation under limited data. Only 1% of the training data is supervised using physics-based loss.

> Code adapted from [Ben Moseley's repository](https://github.com/benmoseley/harmonic-oscillator-pinn) and his blog post:  
> ["So, what is a physics-informed neural network?"](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/)

### `3_HarmonicOscillator`
Not included in the thesis, but compares the **carbon footprint** of a NN and a PINN on a simple harmonic oscillator task.

> Code also adapted from [Ben Moseley's repository](https://github.com/benmoseley/harmonic-oscillator-pinn).

### `4_Burgers`
Data is governed by the viscous Burger's eq. We add **carbon footprint** as an evaluation criterion. Compares:
- A vanilla MLP (no physics inductive bias)
- A baseline PINN trained purely on the governing PDE
- A data-enhanced PINN combining physics and data supervision
- A constraint-enforced PINN with boundary/initial conditions built into the architecture

Models are compared in terms of both predictive accuracy and energy consumption.

> Code adapted from [pinns_tutorial](https://github.com/nguyenkhoa0209/pinns_tutorial):  
> - [Colab notebook 1](https://colab.research.google.com/drive/1na1yVhBF9MYPntbr6bfGd6qwWKl-uJGS?usp=sharing)  
> - [Colab notebook 2](https://colab.research.google.com/drive/1EEDH099GalrgqNbEaDgPm-OCHULBQ3HT?usp=sharing)

### `5_CylinderFlow`
Not used in the thesis. A preliminary exploration of PINNs on a **2D Cylinder Flow** dataset.

### `plot_NNs.ipynb`
Helper notebook for **visualizing neural network architectures**.
