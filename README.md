# Python Learning Notebooks for Scientists

This collection provides comprehensive pytorch and related tutorials. 

## Part 1: Foundations

### 1. [python_basics.ipynb](a_basics/01a_python_basics.ipynb)
Core Python syntax, data types, control flow, functions, classes, and file I/O

### 2. [pandas_basics.ipynb](a_basics/03a_pandas_basics.ipynb)
Data manipulation, DataFrames, indexing, grouping, merging, and time series

### 3. [numpy_basics.ipynb](a_basics/02a_numpy_basics.ipynb)
Array operations, mathematical functions, linear algebra, and performance tips

### 4. [matplotlib_basics.ipynb](a_basics/04a_matplotlib_basics.ipynb)
Essential plotting techniques, customization, subplots, and saving figures

### 5. [scikit_learn_basics.ipynb](a_basics/05a_scikit_learn_basics.ipynb)
Machine learning workflows, classification, regression, clustering, and model comparison

*Each Part 1 notebook is designed for breadth over depth. They include practical examples with scientific data scenarios and cover the most commonly used features of each library.*

## Part 2: Advanced Techniques

### 1. [python_basics_part2.ipynb](a_basics/01b_python_basics_part2.ipynb) - Advanced Python Concepts
- Advanced functions (`*args`, `**kwargs`, decorators, closures)
- Context managers and the `with` statement
- Generators and iterators
- Advanced OOP (abstract classes, dataclasses, protocols)
- Advanced data structures (collections module, enums)
- Metaprogramming and introspection
- Concurrency and performance optimization

### 2. [pandas_basics_part2.ipynb](a_basics/03b_pandas_basics_part2.ipynb) - Advanced Data Analysis
- MultiIndex and hierarchical indexing
- Advanced GroupBy operations and custom aggregations
- Advanced time series analysis and resampling
- Feature engineering and data transformation
- Performance optimization and memory management
- Advanced plotting integration with statistical analysis

### 3. [numpy_basics_part2.ipynb](a_basics/02b_numpy_basics_part2.ipynb) - Advanced Numerical Computing
- Advanced array operations and broadcasting
- Linear algebra decompositions (SVD, PCA, QR, Cholesky)
- Signal processing and Fourier analysis
- Advanced statistical operations and random sampling
- Optimization and root finding with scipy
- Memory optimization and performance techniques

### 4. [matplotlib_basics_part2.ipynb](a_basics/04b_matplotlib_basics_part2.ipynb) - Advanced Visualization
- Complex subplot layouts with GridSpec
- Advanced 3D plotting and surface visualization
- Publication-quality figures and professional styling
- Advanced statistical visualizations
- Interactive plots and widgets
- Animation and dynamic visualizations
- Custom colormaps and advanced styling

### 5. [scikit_learn_basics_part2.ipynb](a_basics/05b_scikit_learn_basics_part2.ipynb) - Advanced Machine Learning
- Advanced preprocessing and feature engineering pipelines
- Hyperparameter tuning with GridSearch and RandomizedSearch
- Feature selection and importance analysis
- Advanced model evaluation and interpretation
- Ensemble methods and model stacking
- Model deployment preparation and production considerations

*Each Part 2 notebook builds significantly on the basics, covering production-ready techniques, advanced workflows, and real-world applications that a scientist would encounter in sophisticated data analysis and machine learning projects.*

## Part 3: Deep Learning with PyTorch

### 1. [pytorch_basics_part1.ipynb](b_pytorch_ipynb/06_pytorch_basics_part1.ipynb) - PyTorch Fundamentals
- Tensor creation, properties, and operations
- Device management (CPU/GPU) and data type handling
- Basic tensor manipulations and mathematical operations
- Indexing, slicing, and reshaping tensors
- Broadcasting and reduction operations
- NumPy integration and data conversion

### 2. [pytorch_basics_part2.ipynb](b_pytorch_ipynb/06_pytorch_basics_part2.ipynb) - Autograd and Neural Networks
- Automatic differentiation with autograd
- Building neural networks with nn.Module
- Common layers (Linear, Conv2d, pooling) and activation functions
- Loss functions and optimizers (SGD, Adam, RMSprop)
- Training loops and gradient management
- Model saving and loading

### 3. [pytorch_basics_part3.ipynb](b_pytorch_ipynb/06_pytorch_basics_part3.ipynb) - Data Loading and Datasets
- Custom Dataset classes and data preprocessing
- DataLoader for efficient batching and shuffling
- Train/validation/test splits with random_split
- Data transformations and augmentation techniques
- Working with CSV files and real datasets
- Data normalization and standardization best practices

### 4. [pytorch_basics_part4.ipynb](b_pytorch_ipynb/06_pytorch_basics_part4.ipynb) - CNNs and Computer Vision
- Convolutional layers and kernel operations
- CNN architectures and building blocks
- Image data preprocessing and augmentation
- Training CNNs for classification tasks
- Transfer learning and pre-trained models
- Feature visualization and model interpretation

### 5. [pytorch_basics_part5.ipynb](b_pytorch_ipynb/06_pytorch_basics_part5.ipynb) - Advanced Topics and Production
- Mixed precision training for efficiency
- Model quantization and optimization techniques
- TorchScript for model serialization and deployment
- Custom autograd functions and CUDA kernels
- Model checkpointing and distributed training
- Production deployment considerations and best practices

### 6. [pytorch_basics_part6.ipynb](b_pytorch_ipynb/06_pytorch_basics_part6.ipynb) - RNNs and Natural Language Processing
- Recurrent Neural Networks (SimpleRNN, LSTM, GRU)
- Text preprocessing and tokenization techniques
- Sequence modeling and language modeling
- Text classification and sentiment analysis
- Handling variable-length sequences
- Word embeddings and their applications

### 7. [pytorch_basics_part7.ipynb](b_pytorch_ipynb/06_pytorch_basics_part7.ipynb) - Transformers and Modern NLP
- Attention mechanisms and scaled dot-product attention
- Multi-head attention and positional encoding
- Building Transformer encoders from scratch
- Working with pre-trained models (BERT, GPT)
- Fine-tuning strategies for NLP tasks
- Hugging Face Transformers integration

### 8. [pytorch_basics_part8.ipynb](b_pytorch_ipynb/06_pytorch_basics_part8.ipynb) - Generative Models and Unsupervised Learning
- Autoencoders and dimensionality reduction
- Variational Autoencoders (VAEs) for probabilistic modeling
- Generative Adversarial Networks (GANs)
- Self-supervised learning techniques
- Contrastive learning and representation learning
- Comparing different generative approaches

### 9. [pytorch_basics_part9.ipynb](b_pytorch_ipynb/06_pytorch_basics_part9.ipynb) - Advanced Architectures and Specialized Domains
- Graph Neural Networks (GNNs) and node classification
- Vision Transformers (ViTs) for image recognition
- Object detection with YOLO-style architectures
- Time series forecasting with deep learning
- Multi-modal learning (combining vision and text)
- Advanced architectural patterns and design principles

### 10. [pytorch_basics_part10.ipynb](b_pytorch_ipynb/06_pytorch_basics_part10.ipynb) - MLOps and Advanced Production
- Experiment tracking and reproducibility
- Model registry and version management
- Model monitoring and drift detection
- A/B testing frameworks for ML models
- Advanced deployment strategies and scaling
- Production pipelines and best practices

*The PyTorch series provides a comprehensive journey from tensor fundamentals through advanced production-ready deep learning systems. Starting with basic concepts and progressing through modern architectures, generative models, and MLOps practices, each notebook includes practical examples with real-world applications relevant to scientific computing and research.*

## Part 4: Graph Neural Networks with PyTorch Geometric

### 1. [pytorch_geometric_part1.ipynb](c_torch_geometric_ipynb/07_pytorch_geometric_part1.ipynb) - Message Passing Networks
- Graph Convolutional Networks (GCN) fundamentals and spectral approaches
- GraphSAGE with neighbor sampling and inductive learning capabilities
- Graph Isomorphism Networks (GIN) for theoretically powerful graph classification
- Graph Attention Networks (GAT) with multi-head attention mechanisms
- Comprehensive comparison on node classification tasks
- CPU optimization strategies for MacBook Air M2

### 2. [pytorch_geometric_part2.ipynb](c_torch_geometric_ipynb/07_pytorch_geometric_part2.ipynb) - Graph Autoencoders
- Graph Autoencoder (GAE) for unsupervised graph representation learning
- Variational Graph Autoencoder (VGAE) with probabilistic latent spaces
- Link prediction tasks and evaluation metrics (AUC, AP)
- Node clustering and community detection applications
- Graph reconstruction quality analysis and visualization
- Memory-efficient training techniques for CPU environments

### 3. [pytorch_geometric_part3.ipynb](c_torch_geometric_ipynb/07_pytorch_geometric_part3.ipynb) - Graph Transformers
- Graph positional encoding strategies (Laplacian, degree-based, learned)
- Multi-head self-attention mechanisms adapted for graph structures
- Complete Graph Transformer (GraphiT) architecture implementation
- Lightweight transformer designs optimized for CPU performance
- Attention pattern analysis and visualization for interpretability
- Comparison with traditional GNNs on node and graph classification

### 4. [pytorch_geometric_part4.ipynb](c_torch_geometric_ipynb/07_pytorch_geometric_part4.ipynb) - Memory-Enhanced GNNs
- GraphSAINT sampling strategies (random walk, node, edge sampling)
- FastGCN with importance sampling for scalable neighbor selection
- Memory usage monitoring and optimization techniques
- Scalability testing on synthetic large graphs
- Performance trade-offs between accuracy and computational efficiency
- CPU-specific optimization guidelines for resource-constrained environments

*The PyTorch Geometric series covers the complete spectrum of Graph Neural Networks, from foundational message passing to cutting-edge transformer architectures and scalable training methods. Each notebook is optimized for CPU training on MacBook Air M2, with comprehensive comparisons, visualizations, and practical optimization strategies for real-world graph learning applications.*

## Part 5: Reinforcement Learning with PyTorch

### 1. [RL_planB_part1.ipynb](d_RL/RL_planB_part1.ipynb) - RL Fundamentals & Tabular Methods
- Markov Decision Processes (MDPs) with complete mathematical treatment
- Bellman equations: derivation, intuition, and practical computation
- Value iteration and policy iteration algorithms with convergence analysis
- Q-learning and SARSA with exploration strategies and convergence theory
- Tabular methods implementation and comparative analysis
- GridWorld environments with visualization and debugging techniques

### 2. [RL_planB_part2.ipynb](d_RL/RL_planB_part2.ipynb) - Monte Carlo & Temporal Difference Learning
- Monte Carlo methods: prediction and control with variance analysis
- Temporal difference learning theory, intuition, and practical implementation
- Eligibility traces and n-step methods for improved learning efficiency
- On-policy vs off-policy learning trade-offs and convergence properties
- Practical convergence analysis and debugging techniques for RL algorithms
- CliffWalking and WindyGridWorld implementations with performance comparisons

### 3. [RL_planB_part3.ipynb](d_RL/RL_planB_part3.ipynb) - From Tabular to Deep RL
- Function approximation: necessity, challenges, and mathematical foundations
- Linear function approximation with feature engineering and theoretical analysis
- Neural networks for value function approximation with stability considerations
- The deadly triad (function approximation + bootstrapping + off-policy) and solutions
- Experience replay: theory, implementation, and empirical benefits
- Target networks and their mathematical justification for training stability

### 4. [RL_planB_part4.ipynb](d_RL/RL_planB_part4.ipynb) - Deep Q-Learning
- Deep Q-Network (DQN) architecture, training procedures, and hyperparameter sensitivity
- Double DQN and overestimation bias mitigation with theoretical analysis
- Dueling DQN network architecture benefits and empirical comparisons
- Prioritized experience replay with importance sampling and bias correction
- Rainbow DQN: combining multiple improvements for state-of-the-art performance
- Implementation best practices, debugging techniques, and performance analysis

### 5. [RL_planB_part5.ipynb](d_RL/RL_planB_part5.ipynb) - Policy Gradient Methods
- Policy gradient theorem: complete mathematical derivation and intuition
- REINFORCE algorithm with baseline reduction and variance analysis
- Actor-Critic methods: theory, implementation, and advantage estimation techniques
- Proximal Policy Optimization (PPO): mathematical foundations and practical implementation
- Continuous vs discrete action spaces with appropriate network architectures
- Policy gradient debugging, hyperparameter tuning, and performance optimization

### 6. [RL_planB_part6.ipynb](d_RL/RL_planB_part6.ipynb) - Advanced Methods & Applications
- Soft Actor-Critic (SAC) for continuous control with entropy regularization
- Model-based RL fundamentals: Dyna-Q integration of planning and learning
- Transfer learning and domain adaptation techniques for RL applications
- Real-world deployment challenges: safety, robustness, and monitoring considerations
- Sample efficiency techniques and comparative analysis across methods
- Current research frontiers: multi-agent RL, meta-learning, and future directions

*The Reinforcement Learning series provides a comprehensive journey from fundamental concepts through state-of-the-art methods. Each notebook includes extensive mathematical exposition with LaTeX, practical implementations optimized for CPU training on MacBook Air M2, and real-world application examples. The series covers both theoretical foundations and practical deployment considerations for RL systems.*

## Part 6: ML Practice Questions Series

A comprehensive collection of machine learning practice questions covering fundamental to advanced concepts. Each notebook follows a structured Q&A format with theoretical explanations, practical implementations, and real-world applications.

### 1. [ML_practice_part1.ipynb](f_ML_practice/ML_practice_part1.ipynb) - ML Fundamentals and Problem Types
- Machine learning paradigms (supervised, unsupervised, reinforcement learning)
- Problem classification (regression, classification, clustering)
- Training, validation, and test set strategies
- Performance metrics selection and interpretation
- Common ML pitfalls and how to avoid them

### 2. [ML_practice_part2.ipynb](f_ML_practice/ML_practice_part2.ipynb) - Data Preprocessing and Feature Engineering
- Missing data handling strategies (imputation techniques, impact analysis)
- Feature scaling and normalization (when to use each method)
- Categorical variable encoding (one-hot, target, ordinal encoding)
- Feature selection techniques (filter, wrapper, embedded methods)
- Data leakage prevention and feature engineering best practices

### 3. [ML_practice_part3.ipynb](f_ML_practice/ML_practice_part3.ipynb) - Model Evaluation and Validation
- Cross-validation strategies for different data types and scenarios
- Bias-variance tradeoff analysis with practical implementations
- Performance metrics for imbalanced datasets and specialized domains
- Statistical significance testing for model comparison
- Model selection frameworks and hyperparameter optimization

*Each ML Practice notebook contains 15-20 focused questions with detailed theoretical explanations, mathematical foundations, practical implementations, and comparative analysis. The series is designed to test and reinforce understanding of key ML concepts through progressively challenging questions.*

## Part 7: ML LeetCode Series

Algorithmic challenges focused on implementing fundamental machine learning algorithms and data structures from scratch. Each problem follows LeetCode format with multiple solution approaches and complexity analysis.

### 1. [ML_leetcode_part1.ipynb](e_ML_leetstyle/ML_leetcode_part1.ipynb) - Linear Algebra and Optimization
- Matrix multiplication optimization (naive, blocked, Strassen algorithms)
- QR decomposition using Householder reflections
- Gradient descent variants (vanilla, momentum, Nesterov, conjugate gradient)
- Singular Value Decomposition (SVD) with Golub-Reinsch algorithm
- Performance benchmarking and complexity analysis

### 2. [ML_leetcode_part2.ipynb](e_ML_leetstyle/ML_leetcode_part2.ipynb) - Core ML Algorithms from Scratch
- K-means clustering with multiple initialization strategies
- Decision tree classifier with information gain and pruning
- Regularized linear regression (Ridge, Lasso, Elastic Net)
- K-nearest neighbors with custom distance metrics
- Algorithm comparison and optimization techniques

### 3. [ML_leetcode_part3.ipynb](e_ML_leetstyle/ML_leetcode_part3.ipynb) - Data Structures and Efficiency
- KD-tree for efficient nearest neighbor search
- Locality Sensitive Hashing (LSH) for approximate similarity search
- Count-Min Sketch for streaming frequency estimation
- Bloom filters for space-efficient set membership testing
- Performance trade-offs and scalability analysis

### 4. [ML_leetcode_part4.ipynb](e_ML_leetstyle/ML_leetcode_part4.ipynb) - Advanced ML Algorithms
- Gaussian Mixture Model with EM algorithm
- XGBoost-style gradient boosting with second-order optimization
- Variational Autoencoder (VAE) with reparameterization trick
- Advanced probabilistic modeling and generative algorithms
- Production-ready implementations with numerical stability

*The ML LeetCode series covers algorithmic challenges from basic linear algebra through advanced machine learning algorithms. Each problem includes multiple solution approaches, complexity analysis, and performance benchmarking against reference implementations. Perfect for technical interviews and deepening algorithmic understanding.*

## Getting Started

1. **Foundations**: Start with Part 1 notebooks for core Python skills
2. **Advanced Techniques**: Progress to Part 2 for production-ready expertise
3. **Deep Learning**: Complete PyTorch series (Part 3) for neural networks
4. **Specialized Domains**: Explore Graph Neural Networks (Part 4) and Reinforcement Learning (Part 5)
5. **Practice & Assessment**: Use ML Practice Questions (Part 6) to test understanding
6. **Technical Mastery**: Challenge yourself with ML LeetCode (Part 7) for algorithmic depth
7. Each notebook is self-contained but builds conceptually on previous ones
8. All examples use scientific scenarios and realistic datasets

## Requirements

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required packages will be imported in each notebook
