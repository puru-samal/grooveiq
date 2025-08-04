# 🥁 GrooveIQ: Spase Representations for Controllable Drum Pattern Generation

## Overview

This repository contains the codebase for my master's thesis. While drumming with nuance takes years to master, most people already have an intuitive sense of rhythm. **GrooveIQ** bridges that gap—it's a deep learning system for interactive drum pattern generation that transforms simple, time-aligned button presses into expressive drum performances. At its core is a novel Variational Autoencoder (VAE) that learns to map between high-dimensional drum sequences and low-dimensional control signals, enabling intuitive, real-time groove creation through sparse rhythmic input.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd grooveiq

# Install dependencies
pip install -r requirements.txt
```

### Training

`grooveiq_runner.ipynb` provides a minimal example for training a model.

### Inference

`grooveiq_analysis.ipynb` provides a minimal example for evaluating a model.
`grooveiq_inference.ipynb` provides several minimal examples for running inference.

### Interactive Demo

```bash
# Run the interactive drum generation demo
python grooveiq_demo.py
```

## 📁 Project Structure

```bash
grooveiq/
├── data/ # Dataset processing/preprocessing and loading
│ ├── dataset.py     # Main dataset class
│ ├── descriptors.py # Feature extraction
│ └── feature.py     # Datatype abstraction for drum features
│ └── ...            # Other dataset processing preprocessing files
├── models/ # Neural network architectures
│ ├── GrooveIQ.py  # Main model implementation
│ └── sub_modules/ # Model components
├── trainers/ # Training infrastructure
│ ├── base_trainer.py     # Base training class
│ └── grooveiq_trainer.py # GrooveIQ-specific trainer
├── utils/ # Various utility functions
│ └── ...
├── grooveiq_demo.py # Interactive demo
├── grooveiq_analysis.ipynb  # Analysis notebook
├── grooveiq_inference.ipynb # Inference notebook
├── grooveiq_runner.ipynb    # Training notebook
└── requirements.txt # Dependencies
```
