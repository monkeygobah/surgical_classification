# Classification of Surgeon Skill Level using V-JEPA and LSTM Models

## Overview
This repository contains code for classifying surgeon skill levels based on surgical videos. The project utilizes the Cataract 101 dataset, leveraging both the V-JEPA model and an LSTM model with a ResNet-18 feature extractor.

## Key Features
1. **Dataset: Cataract 101**:
   - The dataset is split into various surgical phases as part of preprocessing to enhance the accuracy of classification.

2. **Binary Classification of Surgeon Skill Level**:
   - The project focuses on binary classification of surgeon skill level based on surgical videos.

3. **V-JEPA Model**:
   - Utilizes the V-JEPA model released by Meta in 2024.
   - Modified data loading to select frames from the middle of the video.
   - Trained an attention probe on top of the pre-trained backbone from Meta to improve classification performance.

4. **LSTM Model**:
   - Used ResNet-18 pretrained on ImageNet as a feature extractor.
   - Employed LSTM layers downstream to make predictions on the surgical videos.

5. **Experiments Conducted**:
   - Various experiments were conducted, including:
     - Training and testing on easy and hard surgical phases.
     - Training and testing on each surgical phase individually.
     - Training and testing on different combinations of surgical phases.

## Getting Started
To get started with this project, follow these steps:

### Prerequisites
- Python 3.10+
- PyTorch
- [V-JEPA Model](https://github.com/facebookresearch/vjepa)
- [Cataract 101 Dataset](https://example.com/cataract101)

### NOTES
1. See V-JEPA for installation guidance, but can use cataract config yaml in evals to replicate experiments
2. See  https://docs.google.com/document/d/1uCtWCGGY4TMgrgtyK3U0uNzPn4oZ1f5jL-aDRxwgxaE/edit?usp=sharing for more details of this project

Collaborators: Mahtab Faraji, Saaketh Kosaraju
