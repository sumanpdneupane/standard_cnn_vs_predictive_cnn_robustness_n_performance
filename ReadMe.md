# Performance and Robustness Analysis of Standard CNN and Predictive Coding CNN under Controlled Noise Perturbations

# Problem Trying to Solve
```
Deep learning models, particularly Convolutional Neural Networks (CNNs), have achieved 
remarkable accuracy on image classification tasks. However, standard CNNs trained with 
backpropagation are known to be fragile when exposed to noisy or corrupted inputs. This 
becomes a critical limitation in real-world deployments where images are rarely perfectly 
clean.

Predictive Coding Networks (PCNs), inspired by how the human brain processes visual 
information, offer a biologically plausible alternative learning mechanism. Instead of 
global gradient propagation, PCNs rely on local prediction error minimization to update 
representations. Despite their theoretical advantages, PCNs have not been systematically 
evaluated against standard CNNs in terms of robustness under controlled noise conditions 
across benchmark datasets.

This research is grounded in both foundational and recent developments in predictive 
coding:
•	Rao & Ballard, Predictive Coding in the Visual Cortex, 1999
•	PrediRep: Modelling Hierarchical Predictive Coding with an Unsupervised Deep Learning 
    Network, 2022
•	Brain-inspired predictive coding dynamics improve the robustness of deep neural 
    networks, 2023
•	Hosseini, Hierarchical Predictive Coding Models in a Deep-Learning Framework, 2023
•	Benchmarking Predictive Coding Networks - Made Simple, 2024
•	Stenlund, Introduction to PCN for Machine Learning, 2025
•	Salvatori, Towards the Training of Deeper Predictive Coding Neural Networks, 2025
```

# Why Predictive Coding?
```
Because standard CNNs rely on a single forward pass and are sensitive to noise, whereas 
predictive coding uses iterative error correction, which may improve robustness under 
noisy conditions.
```

# Research Question
```
Based on the identified research gap, the following research questions are formulated:
1.  How do Standard CNN and PC-CNN compare in baseline performance on 
    clean datasets?
2.  How do Standard CNN and PC-CNN compare in robustness under Gaussian 
    noise perturbations?
3.  How do Standard CNN and PC-CNN compare in robustness under Salt & Pepper 
    noise perturbations?
```

# Methodology- How Research Solves It
```
This research implements and compares two models:
•	Standard CNN trained using backpropagation
•	PC-CNN trained using local predictive coding principles with prediction error 
minimization

The comparison will be conducted across two benchmark datasets of increasing complexity: 
•	MNIST  
•	CIFAR-10. 

Two main experiments will be performed:
•	Experiment 1- Performance: Both models will be evaluated on clean data to establish 
    baseline accuracy, training efficiency, and convergence behavior.

•	Experiment 2- Robustness: Both models will be tested under controlled noise perturbations 
    including Gaussian noise and Salt & Pepper noise at multiple controlled noise levels to 
    analyze performance degradation patterns under increasing noise conditions.

This research investigates the comparative performance and robustness of Standard CNN and PC-CNN 
under controlled noise conditions, examining whether the iterative top-down inference mechanism 
of PC-CNN offers advantages over the single forward pass used in standard CNNs.
```

# Experimental Design
```
To ensure a structured and scientifically valid comparison, the experimental design is defined as follows:
```
### Evaluation Metrics
```
The following metrics will be used to evaluate model performance:
•   Accuracy: Overall classification correctness
•   Robustness Score: Rate of performance degradation under increasing noise levels
•   Convergence Speed: Number of epochs required to reach stable performance
•   Loss Function Behavior: Training stability and optimization efficiency

Additionally, performance degradation curves (Accuracy vs Noise Level) will be plotted 
to compare robustness between models.
```

### Experimental Fairness
```
To ensure a fair comparison between Standard CNN and PC-CNN:
•   Both models will use identical architectures (where applicable)
•   Same datasets and preprocessing steps will be applied
•   Training hyperparameters (epochs, batch size, learning rate) will be aligned 
    as closely as possible
•   Evaluation will be conducted under identical noise conditions

This ensures that differences in performance are attributable to learning mechanisms 
rather than experimental bias.
```

### Expected Outputs
```
•   Accuracy vs Noise Level graphs for both Gaussian and Salt & Pepper noise
•   Comparative performance tables across all noise levels
•   Visualization of noisy inputs across datasets
•   Analysis of performance degradation patterns

These outputs will provide both quantitative and visual evidence of model robustness.
```

# Languages, Frameworks and Tools
```
Tool            Purpose
Python          3.x Primary programming language
PyTorch         Deep learning framework for both models
TorchVision     Dataset loading for MNIST and CIFAR-10
NumPy           Numerical computations
Matplotlib      Result visualization and plotting
Training        Google Colab (NVIDIA T4) / MacBook M3 Pro (18GB) Model 
                training and experimentation
GitHub          Version control and code sharing
```

# High Computation Requirement
```
Two hardware environments will be used: a MacBook M3 Pro (18GB, MPS backend) for debugging 
and smaller experiments, and Google Colab T4 GPU (CUDA) for larger experiments and final 
runs, with actual running times varying depending on dataset size, hyperparameters, and 
hardware performance. 
```

# Contribution / Novelty - How This Research is Unique
```
Globally: 
Predictive Coding Networks are an emerging research area with key papers published 
as recently as 2025. While individual PCN implementations exist, a systematic robustness 
comparison between standard CNNs and convolutional PC-CNNs across multiple datasets under 
controlled noise perturbations has not been comprehensively published. This research addresses 
that gap by providing empirical evidence to explore whether biologically plausible learning 
rules can offer robustness advantages compared to backpropagation.

In the Context of Nepal: 
Deep learning research in Nepal is still developing, with most 
academic work focusing on applying existing models rather than investigating fundamental 
learning mechanisms. This research explores biologically inspired alternatives to 
backpropagation, specifically Predictive Coding, a neuroscience-grounded framework for 
image classification. It contributes to positioning Nepali AI research within global 
discussions on next-generation learning algorithms, neuromorphic computing, and 
brain-inspired artificial intelligence. 

The findings may also inform future research on energy-efficient AI systems, which is 
particularly relevant for resource-constrained computing environments.
```