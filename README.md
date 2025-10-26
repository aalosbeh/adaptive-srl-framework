# Adaptive Multi-Modal AI Framework for Self-Regulated Learning in Generation Z

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A federated deep reinforcement learning framework for personalized self-regulated learning support with multi-modal learning analytics and privacy preservation.

## Overview

This repository contains the implementation of the Adaptive Multi-Modal AI Framework described in the paper "Adaptive Multi-Modal AI Framework for Personalized Self-Regulated Learning in Generation Z: A Federated Deep Reinforcement Learning Approach" by Anas AlSobeh, Amani Shatnawi, and Ahmad Asfour.

The framework integrates:
- **Multi-modal learning analytics** (text, visual, temporal, graph)
- **Federated deep reinforcement learning** for collaborative model training
- **Hierarchical attention mechanisms** for metacognitive state estimation
- **Privacy-preserving techniques** (differential privacy, secure aggregation)
- **Personalized intervention generation** for SRL support

## Key Features

 **Privacy-Preserving**: Federated learning with differential privacy guarantees
 **Multi-Modal**: Integrates text, visual, temporal, and graph data
 **Personalized**: Adaptive interventions based on learner state
 **Metacognitive**: Real-time estimation of awareness, monitoring, and control
 **Scalable**: Distributed architecture for multiple institutions
 **Research-Ready**: Comprehensive evaluation metrics and baselines

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher
- CUDA 11.0+ (optional, for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/adaptive-srl-framework.git
cd adaptive-srl-framework

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### 1. Generate Synthetic Data

```python
from src.data.synthetic_generator import SyntheticDataGenerator

# Initialize data generator
generator = SyntheticDataGenerator(
    num_institutions=10,
    num_learners=1000,
    num_days=180
)

# Generate dataset
dataset = generator.generate()
dataset.save('data/synthetic_dataset.pkl')
```

### 2. Train the Model

```python
from src.training.federated_trainer import FederatedTrainer
from src.models.srl_framework import SRLFramework

# Initialize framework
framework = SRLFramework(config='configs/default.yaml')

# Initialize federated trainer
trainer = FederatedTrainer(
    framework=framework,
    num_rounds=50,
    local_epochs=5
)

# Train the model
trainer.train(dataset)
```

### 3. Evaluate Performance

```python
from src.evaluation.evaluator import Evaluator

# Initialize evaluator
evaluator = Evaluator(framework)

# Run comprehensive evaluation
results = evaluator.evaluate(test_dataset)
evaluator.generate_report(results, output_dir='results/')
```

## Architecture

The framework consists of several key components:

### 1. Multi-Modal Data Processing
- **Text Encoder**: BERT-based transformer for textual data
- **Visual Encoder**: ResNet + LSTM for visual engagement
- **Temporal Encoder**: Dilated CNN for behavioral sequences
- **Graph Encoder**: Graph Attention Networks for relational data

### 2. Metacognitive State Estimation
- **Hierarchical Attention**: Dynamic modality weighting
- **State Estimators**: Awareness, monitoring, and control prediction
- **Real-time Processing**: Continuous state updates

### 3. Federated Deep Reinforcement Learning
- **Local Training**: PPO-based policy optimization
- **Federated Aggregation**: Privacy-preserving parameter updates
- **Global Model**: Collaborative learning across institutions

### 4. Intervention Generation
- **Content Recommendations**: Personalized learning materials
- **Strategy Suggestions**: Metacognitive strategy guidance
- **Feedback Delivery**: Adaptive feedback mechanisms
- **Social Facilitation**: Collaborative learning support

## Project Structure

```
adaptive-srl-framework/
├── src/
│   ├── data/              # Data generation and loading
│   ├── models/            # Neural network architectures
│   ├── training/          # Training loops and optimization
│   ├── evaluation/        # Evaluation metrics and analysis
│   ├── privacy/           # Privacy preservation mechanisms
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── data/                  # Dataset storage
├── models/                # Trained model checkpoints
├── examples/              # Example scripts and notebooks
├── requirements.txt       # Python dependencies
├── setup.py               # Package installation script
└── README.md              # This file
```

## Configuration

The framework uses YAML configuration files. Example configuration:

```yaml
model:
  text_encoder:
    model_name: "bert-base-uncased"
    hidden_size: 768
  visual_encoder:
    cnn_backbone: "resnet50"
    lstm_hidden: 512
  attention:
    num_heads: 8
    dropout: 0.1

training:
  federated:
    num_rounds: 50
    local_epochs: 5
    learning_rate: 0.001
  privacy:
    epsilon: 1.0
    delta: 1e-5
    noise_multiplier: 1.1

intervention:
  types:
    - content
    - strategy
    - feedback
    - social
```

## Experiments

To reproduce the experiments from the paper:

```bash
# Run all experiments
python experiments/run_all.py

# Run specific experiment
python experiments/metacognitive_estimation.py
python experiments/federated_learning.py
python experiments/intervention_effectiveness.py
python experiments/privacy_analysis.py
```

## Results

Our framework achieves:
- **91.2%** metacognitive state estimation accuracy
- **85%** intervention effectiveness
- **92%** overall system performance
- **95%** privacy preservation score

See the [paper](docs/paper.pdf) for detailed results and analysis.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{alsobeh2025adaptive,
  title={Adaptive Multi-Modal AI Framework for Personalized Self-Regulated Learning in Generation Z: A Federated Deep Reinforcement Learning Approach},
  author={AlSobeh, Anas and Shatnawi, Amani and Asfour, Ahmad},
  journal={International Journal of Educational Technology},
  year={2025}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This research was conducted at Southern Illinois University and Weber State University. We thank all contributors and reviewers for their valuable feedback.

## Contact

- Anas AlSobeh - anas.alsobeh@siu.edu
- Amani Shatnawi - amanishatnawi1@weber.edu
- Ahmad Asfour - aasfour@weber.edu

## Support

For questions and issues, please open an issue on GitHub or contact the authors directly.

