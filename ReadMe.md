# Performance and Robustness Analysis of Standard CNN and Predictive Coding CNN under Controlled Noise Perturbations

# Problem Trying to Solve
```
Deep learning models, particularly Convolutional Neural Networks (CNNs), have achieved remarkable accuracy on 
image classification tasks. However, standard CNNs trained with backpropagation are known to be fragile when 
exposed to noisy or corrupted inputs. This becomes a critical limitation in real-world deployments where images 
are rarely perfectly clean.

Predictive Coding Networks (PCNs), inspired by how the human brain processes visual information, offer a 
biologically plausible alternative learning mechanism. Instead of global gradient propagation, PCNs rely 
on local prediction error minimization to update representations.

Despite their theoretical advantages, PCNs have not been systematically evaluated against standard CNNs in 
terms of robustness under controlled noise conditions across benchmark datasets.

This research is grounded in both foundational and recent developments in predictive coding:
1. Rao & Ballard, Predictive Coding in the Visual Cortex, 1999
2. PrediRep: Modelling Hierarchical Predictive Coding with an Unsupervised Deep Learning Network, 2022
3. Brain-inspired predictive coding dynamics improve the robustness of deep neural networks, 2023
4. Hosseini, Hierarchical Predictive Coding Models in a Deep-Learning Framework, 2023
5. Benchmarking Predictive Coding Networks — Made Simple, 2024
6. Stenlund, Introduction to PCN for Machine Learning, 2025
7. Salvatori, Towards the Training of Deeper Predictive Coding Neural Networks, 2025
```

# How Research Solves It
```
This research implements and compares two models:
1. Standard CNN trained using backpropagation
2. PC-CNN trained using local predictive coding principles with prediction error minimization

The comparison is conducted across two benchmark datasets of increasing complexity: MNIST and CIFAR-10.

Two main experiments will be performed:
Experiment 1 — Performance:
Both models are evaluated on clean data to establish baseline accuracy, training efficiency, and 
convergence behavior.

Experiment 2 — Robustness:
Both models are tested under controlled noise perturbations including Gaussian noise and Salt & Pepper 
noise at multiple controlled noise levels to analyze performance degradation patterns under increasing 
noise conditions.

This research investigates which model demonstrates better performance and robustness under noisy 
conditions, examining whether the iterative top-down inference mechanism of PC-CNN offers advantages 
over the single forward pass used in standard CNNs.
```

# Languages, Frameworks and Tools
```
Tool            Purpose
Python          3.x Primary programming language
PyTorch         Deep learning framework for both models
TorchVision     Dataset loading for MNIST and CIFAR-10
NumPy           Numerical computations
Matplotlib      Result visualization and plotting
Training        Google Colab (NVIDIA T4) / MacBook M3 Pro (18GB) Model training and experimentation
GitHub          Version control and code sharing
```

# How This Research is Unique
```
Globally:
Predictive Coding Networks are an emerging research area with key papers published as recently as 2025. 
While individual PCN implementations exist, a systematic robustness comparison between standard CNNs and 
convolutional PC-CNNs across multiple datasets under controlled noise perturbations has not been 
comprehensively published. This research addresses that gap by providing empirical evidence to explore 
whether biologically plausible learning rules can offer robustness advantages compared to backpropagation.

In the Context of Nepal:
Deep learning research in Nepal is still developing, with most academic work focusing on applying existing 
models rather than investigating fundamental learning mechanisms. This research explores biologically inspired 
alternatives to backpropagation, specifically Predictive Coding, a neuroscience-grounded framework for image 
classification. It contributes to positioning Nepali AI research within global discussions on next-generation 
learning algorithms, neuromorphic computing, and brain-inspired artificial intelligence. The findings may also 
inform future research on energy-efficient AI systems, which is particularly relevant for resource-constrained 
computing environments.
```

# High Computation Requirement
```
The research involves training and comparing Standard CNN and Predictive Coding CNN across two datasets: 
MNIST and CIFAR-10, along with multiple noise experiments. PC-CNN is computationally heavier due to its 
iterative inference loop that runs multiple steps per batch, making it significantly slower per epoch 
than a standard CNN.

Two hardware environments will be used:
1. MacBook M3 Pro (18GB, MPS backend) for debugging and smaller experiments
2. Google Colab T4 GPU (CUDA) for larger experiments and final runs

Actual running times may vary depending on dataset size, hyperparameters, and hardware performance. The 
research is designed to remain feasible within available computational resources through careful experiment 
design and controlled experimental configurations.
```