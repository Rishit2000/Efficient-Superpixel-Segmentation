---
layout: default
title: Project Proposal
permalink: /proposal/
---

# Project Proposal

## 1. Introduction/Background

Semantic segmentation for autonomous driving requires accurate, real-time identification of road elements, vehicles, and pedestrians. However, current state-of-the-art Vision Transformers (ViTs) are often **computationally prohibitive** for on-device deployment. Standard ViTs demand quadratic attention over nearly 10⁶ pixels per image, which is **computationally prohibitive and expensive**.

Even variants that restrict attention to fixed windows compromise fine-grained pixel-level detail, limiting their effectiveness in dense scene understanding. Recent advances show that learned superpixel generation can enhance segmentation by grouping pixels into semantically coherent regions, yet these methods introduce additional training overhead unsuitable for **resource-constrained devices.**

### Literature Review
Real-time semantic segmentation research has evolved through several computational efficiency approaches. The SLIC algorithm [1] established efficient superpixel generation using k-means clustering in five-dimensional color-spatial space, achieving O(N) complexity while maintaining superior boundary adherence. Barcelos et al. [2] presented a comprehensive taxonomy for superpixel segmentation, classifying 23 strategies based on connectivity, compactness, and computational efficiency.

Transformer-based approaches have revolutionized the field through innovative architectures. Mei et al. [3] introduced SPFormer, employing content-adaptive superpixels to replace fixed Vision Transformer patches, achieving 1.4% improvement over DeiT-T through Superpixel Cross Attention mechanisms. The model demonstrates remarkable zero-shot transferability to unseen datasets and inherent explainability through semantic pixel grouping. Zhu et al. [4] developed Superpixel Transformers that decompose pixel space into low-dimensional superpixel representations, demonstrating 32× dimensional reduction while maintaining 15.3 FPS on Cityscapes. Their approach addresses the quadratic complexity of standard attention mechanisms by operating on superpixel tokens rather than individual pixels. Ye et al. [5] proposed ESPNet, integrating ConvNeXt with learnable superpixel algorithms to preserve object boundaries while achieving efficient segmentation performance on high-resolution imagery.

Despite these advances like SPFormer, learned superpixel methods introduce additional parameters and training overhead, making them unsuitable for resource-constrained platforms. A practical gap remains between high-accuracy transformer methods and lightweight solutions deployable in real-time.

### Dataset
We are using **Cityscapes dataset** ([Cityscapes](https://www.cityscapes-dataset.com/)), a collection of high-resolution urban street scenes from **50 cities**, with pixel-level semantic annotations across **30 classes**. It contains a diverse set of stereo video sequences from street scenes, in the form of **5,000 finely annotated frames** and **20,000 coarsely annotated frames**.


## 2. Problem Definition
We identify a clear gap where there is no practical solution that balances the **accuracy of semantic segmentation** with the **efficiency required for real-time, on-device deployment** in autonomous driving contexts.

### Problem
The problem we aim to solve is enabling **fast, lightweight, and accurate semantic segmentation** through **unsupervised superpixel clustering**, bridging the gap between high-accuracy computationally demanding attention operations, and lightweight clustering approaches.

### Motivation
To address this, we combine **unsupervised, parameter-free superpixel generation methods** with **attention based segmentation** that reduce the effective number of image elements without sacrificing segmentation detail. By clustering pixels into compact, perceptually uniform regions, we can dramatically reduce the computational burden of subsequent semantic segmentation while preserving fine object boundaries.

## 3. Methods

We propose a **hybrid pipeline** that combines **unsupervised superpixel clustering** with **supervised semantic segmentation** for efficient, real-time object detection on the Cityscapes dataset.

### Data Preprocessing

- **Image normalization and intensity regularization** to standardize lighting conditions across scenes.  
- **Missing pixel interpolation** to correct corrupted frames and ensure temporal consistency.  
- **Superpixel-level feature extraction** using lightweight CNN encoders (e.g., `torchvision.models.resnet18`) to embed local texture and color information before clustering.  

### Machine Learning Algorithms

- **Unsupervised Learning Models:**  
  Clustering approaches like **SLIC superpixels** (`skimage.segmentation.slic`), **watershed segmentation** (`skimage.segmentation.watershed`), and **graph-cut methods** (`networkx`, `skimage.graph.cut_normalized`) will be evaluated to group pixels into compact, semantically coherent regions.  
- **Supervised Learning Models:**  
  Superpixel features will be generated using **CNNs** (like **ResNets**). We integrate **attention-based layers** (cross-attention modules in `torch.nn.MultiheadAttention`) for segmentation using superpixel features. We'll use lightweight **MLP heads** (`torch.nn.Linear`) for classification.
- **Benchmarking:**  
  Model efficiency will be quantified in terms of **accuracy, parameter count, and inference time**.

This combination leverages the **efficiency of unsupervised clustering** with the **discriminative power of supervised networks**, targeting lightweight and deployable semantic segmentation.

## 4. Potential Results and Discussion
### Quantitative Metrics
- **Mean Intersection over Union (mIoU):** Evaluates overlap between predicted regions and ground-truth segments.  
- **Pixel Accuracy (PA):** Measures overall proportion of correctly classified pixels.  
- **Boundary F1 Score (BF Score):** Assesses how well predicted boundaries align with true object edges.  
- **Inference Efficiency:** Time per image or FPS on target hardware, reflecting real-time capability.  
- **Model Size / Parameter Count:** Ensures feasibility for on-device deployment.  


### Project Goals
- Develop a lightweight, unsupervised superpixel-based segmentation approach suitable for real-time object detection in driving scenarios.  
- Prioritize sustainability by minimizing computational resources and energy consumption.

### Expected Results
- We expect to achieve improved segmentation quality over naïve pixel-wise baselines, with coherent regions, sharper boundaries, and enhanced boundary delineation. 
- The model should maintain reasonably high pixel accuracy and mIoU while remaining lightweight and fast enough for real-time inference. 
- Overall, the approach should demonstrate a practical trade-off between **accuracy, efficiency, and deployability**, highlighting the value of unsupervised superpixels in time-sensitive applications.


## 5. References

[1] R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua, and S. Susstrunk, “SLIC superpixels compared to state-of-the-art superpixel methods,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 11, pp. 2274-2282, Nov. 2012.

[2] I. B. Barcelos, F. C. Belem, L. de M. Joao, Z. K. G. Patrocinio, A. X. Falcao, and S. J. F. Guimaraes, "A Comprehensive Review and New Taxonomy on Superpixel Segmentation," ACM Computing Surveys, vol. 56, no. 8, article 191, Apr. 2024.

[3] J. Mei, L.-C. Chen, A. Yuille, and C. Xie, "SPFormer: Enhancing Vision Transformer with Superpixel Representation," arXiv preprint arXiv:2401.02931, 2024.

[4] A. Z. Zhu, J. Mei, S. Qiao, H. Yan, Y. Zhu, L.-C. Chen, and H. Kretzschmar,"Superpixel Transformers for Efficient Semantic Segmentation,"IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Detroit, MI, USA, 2023, pp. 7651-7658, doi: 10.1109/IROS55552.2023.10341519, 2023.

[5] Z. Ye, Y. Lin, B. Dong, X. Tan, M. Dai, and D. Kong, "An Object-Aware Network Embedding Deep Superpixel for Semantic Segmentation of Remote Sensing Images," Remote Sensing, vol. 16, no. 20, article 3805, Oct. 2024.



## 6. Gantt Chart and Contribution Table

[Click here to View Gantt Chart](https://docs.google.com/spreadsheets/d/1dmYPobnCBGdGa8SsKfX9EAEbnZ38tafuJvTAW1FPgns/edit?usp=sharing)  

### Contribution Table

| Name          | Proposal Contributions                                                                       |
|---------------|----------------------------------------------------------------------------------------------|
| Abhishek      | Maintain GitHub Proposal Page, Literature Review, Gantt Chart, Video                         |
| Amandeep      | Problem Statement, Data-Preprocessing Methodology, Literature Review, Video                  |
| Harshavardhan | Dataset, Supervised segmentation approach, Literature Review, Video                          |
| Kanhaiya      | Methodology, Unsupervised clustering approach, Literature Review, Video                      |
| Rishit        | Project Goals, Metrics, Expected Results, Literature Review, Presentation Slides, Video      |

## 7. GitHub Repository

View our repository here: [GitHub Repository](https://github.gatech.edu/rpatel917/Efficient_Superpixel_Segmentation)

## 8. Project Award Eligibility
We would like to **opt-in** for consideration for the *“Outstanding Project”* award.
