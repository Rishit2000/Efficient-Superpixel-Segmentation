# Efficient Superpixel Segmentation

**CS-7641 Machine Learning Project**
Georgia Institute of Technology

This repository implements a novel approach for efficient semantic segmentation using decoupled superpixel clustering with Vision Transformers (ViT). The project addresses the computational challenges of standard ViTs by combining unsupervised superpixel clustering (SLIC) with transformer-based semantic segmentation.

## Project Structure

### Root Directory Files

**README.md**: Main project documentation and overview

**pyproject.toml**: Python project configuration and dependency management using modern PEP 517/518 standards

**requirements.txt**: Python package dependencies (torch, torchvision, scikit-image, timm, etc.)

### /src/: Source Code Directory

Main implementation directory containing all model code and utilities.

**src/trainer.py**: Main training script with AdamW optimizer, training/validation loops, and checkpoint management. Implements differential learning rates for backbone vs decoder, and includes metrics computation (mIoU, pixel accuracy, confusion matrix).

#### /src/models/: Model Architecture Components

Neural network modules implementing the decoupled superpixel segmentation pipeline.

**src/models/**init**.py**: Package initialization for models module

**src/models/decoupled_superpixel_vit.py**: Main end-to-end model combining all pipeline components (backbone, tokenizer, classifier, associator) into a single DecoupledSuperpixelViT module

**src/models/feature_extractor.py**: CNN backbone (ResNet-based) for extracting dense pixel-level features with configurable output stride and optional backbone freezing

**src/models/superpixel_tokenizer.py**: SLIC-based superpixel generation and feature pooling to convert dense pixel features into compact superpixel tokens

**src/models/superpixel_classifier.py**: Vision Transformer encoder for processing superpixel tokens with multi-head self-attention and MLP layers for semantic classification

**src/models/superpixel_associator.py**: Association module for mapping superpixel-level predictions back to dense pixel-level segmentation masks

#### /src/data/: Dataset and Data Loading

Data loading and preprocessing utilities.

**src/data/**init**.py**: Package initialization for data module

**src/data/dataset_loader.py**: Cityscapes dataset loader with preprocessing, label mapping (from 34 to 19 classes), and augmentation transforms (resize, normalization)

### /dataset/: Dataset Storage

Directory for storing training and validation datasets.

#### /dataset/Cityscapes/: Cityscapes Dataset

Semantic segmentation dataset with urban street scenes.

**dataset/Cityscapes/README.md**: Instructions for downloading and setting up the Cityscapes dataset, including directory structure requirements

### /docs/: Documentation and Reports

Project documentation, proposals, and progress reports.

**docs/index.md**: Project homepage with team information and navigation for the Jekyll website

**docs/proposal.md**: Initial project proposal describing problem definition, background literature, methodology, and dataset

**docs/midterm.md**: Midterm progress report with preliminary results and implementation updates

## Key Features

- **Efficient Segmentation**: Reduces computational complexity from ~10^6 pixels to ~2048 superpixel tokens
- **Unsupervised Clustering**: Uses parameter-free SLIC algorithm for superpixel generation
- **Transformer Architecture**: Multi-head self-attention on superpixel tokens for semantic understanding
- **Dense Predictions**: Association module maps superpixel classifications back to pixel-level predictions
- **Cityscapes Dataset**: Trained on 19-class urban scene segmentation task

## Model Pipeline

1. **Feature Extraction** (PixelFeatureExtractor): ResNet backbone extracts dense features
2. **Superpixel Tokenization** (SuperpixelTokenization): SLIC clustering + feature pooling
3. **Classification** (SuperpixelClassifier): Vision Transformer processes superpixel tokens
4. **Association** (SuperpixelAssociation): Maps predictions back to dense pixel space

## Team

- Abhishek Dharmadhikari
- Amandeep Gupta
- Harshavardhan Kedarnath
- Kanhaiya Madaswar
- Rishit Patel

## Project Website

View the complete project documentation at the GitHub Pages site [here](https://github.gatech.edu/pages/rpatel917/Efficient_Superpixel_Segmentation).
