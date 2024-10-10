Certainly! Here's the README content in a format that you can directly copy and paste into a .md file:

```markdown
# Math Nano GPT: Democratizing Contextual Reasoning in Mathematical Frameworks

## Author: Rupert Tawiah-Quashie
## Hampshire College, May 2024

## Table of Contents
1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Background and Motivation](#background-and-motivation)
4. [Research Questions](#research-questions)
5. [Methodology](#methodology)
   - [Data Collection and Preparation](#data-collection-and-preparation)
   - [Model Architecture](#model-architecture)
   - [Training Process](#training-process)
   - [Fine-tuning](#fine-tuning)
   - [Evaluation Metrics](#evaluation-metrics)
6. [Experiments and Results](#experiments-and-results)
   - [Preliminary Experiments](#preliminary-experiments)
   - [Model Resizing and Training](#model-resizing-and-training)
   - [Fine-tuning Results](#fine-tuning-results)
7. [Discussion](#discussion)
8. [Conclusion](#conclusion)
9. [Future Work](#future-work)
10. [Installation and Usage](#installation-and-usage)
11. [Project Structure](#project-structure)
12. [Acknowledgments](#acknowledgments)
13. [References](#references)
14. [License](#license)
15. [Contact](#contact)

## Abstract

This research investigates techniques for distilling and replicating the contextual reasoning capabilities of large language models into compact, efficient architectures. Motivated by the immense computational requirements of state-of-the-art models, which limit their accessibility, this study aims to develop Math Nano GPT - a lightweight model capable of mathematical reasoning tasks. The research employs a novel approach combining attention mechanisms, sparsity, distillation, and prompt programming to create a specialized architecture. Using the MathPile dataset for pre-training and the Orca-Math dataset for fine-tuning, the study explores self-supervised learning and knowledge transfer from larger models like GPT-4 Turbo.

[[Link to Full Thesis Document](https://drive.google.com/file/d/1QtU5S0kVeSRc-SjBnxEYoWp_ITixyLtw/view)]

## Introduction

The emergence of foundation models like GPT-3 has showcased the immense potential of in-context learning (ICL) to revolutionize artificial intelligence. However, the widespread deployment of ICL in real-world applications is hindered by the massive computational requirements and carbon footprints associated with giant language models. This research addresses this challenge by investigating techniques to distill and replicate the essence of contextual reasoning into far more compact and efficient neural network architectures.

## Background and Motivation

The project is motivated by the need to democratize access to powerful language models capable of complex reasoning tasks. Current state-of-the-art models, while impressive in their capabilities, are often inaccessible due to their computational demands. This research aims to bridge this gap by creating efficient, specialized models for mathematical reasoning.

## Research Questions

1. What key features and structures in large language models help them understand context, reason by analogy, abstract concepts, and generalize across different situations?
2. How can novel model compression techniques and specialized pre-training objectives be developed to ingrain the same capabilities into highly optimized student architectures that are orders of magnitude smaller in size?
3. To what extent can prompt-based knowledge distillation be used to transfer additional analytical skills from larger general-purpose teacher models to size-optimized student models?
4. How can the precision of in-context learning transfer be measured and enhanced using few-shot evaluations, and what iterative improvements can be made to make the transfer more accurate and reliable?
5. How can the versatility and adaptability of large language models be retained in efficient student models, enabling them to tackle a broad range of analytical problems?
6. What theoretical insights can be gained into the underlying mechanics behind how contextual learning and reasoning emerge in neural networks trained under different objectives?

## Methodology

### Data Collection and Preparation

- **Datasets**: 
  - Pre-training: MathPile dataset (Wang et al., 2023)
  - Fine-tuning: Microsoft Orca-Math-Word-Problems-200k dataset (Mitra et al., 2024)
- **Preprocessing**: Custom tokenization process, handling of LaTeX formatting
- **Data loading**: Efficient streaming approach to manage large-scale datasets

### Model Architecture

- Based on the nanoGPT framework (Karpathy, 2022)
- Key components:
  - Multi-head self-attention mechanism
  - Post-norm configuration
  - Gaussian Error Linear Unit (GELU) activation
  - Flash Attention for efficient computation

### Training Process

- **Hardware**: Single NVIDIA A10 GPU
- **Optimization**: 
  - AdamW optimizer
  - Learning rate scheduling
  - Gradient clipping
  - Mixed precision training
- **Hyperparameters**: 
  - Batch size: 1
  - Initial learning rate: 3e-5
  - Number of epochs: 40 (initial training)

### Fine-tuning

- Adaptation to mathematical word problems using the Orca-Math dataset
- Knowledge distillation from GPT-4 Turbo

### Evaluation Metrics

- Primary metric: Validation loss
- Secondary evaluations:
  - Generated output coherence
  - Performance on unseen mathematical tasks
  - Comparison with larger models on benchmark datasets

## Experiments and Results

### Preliminary Experiments

- Initial model: 124.4 million parameters
- Training duration: 12h 26m 7s
- Results:
  - Training loss: 1.803
  - Best validation loss: 2.8986
- Observations: Overfitting occurred after 13 epochs

### Model Resizing and Training

- Hyperparameter sweeps conducted (44 sweeps, 16 hours)
- Ideal configuration (based on sweeps):
  - 2 batches
  - Learning rate: 0.0000653
  - 40+ epochs
  - 19 attention heads
  - 21 layers
  - Vocabulary size: 70,469
- Final model configuration:
  - Parameters: 751 million
  - Batch size: 1
  - Learning rate: 0.00003
  - 40 epochs
  - 18 attention heads
  - 16 layers

#### Training Results for 751M Model

- Training duration: 2 days, 18 hours, 42 minutes
- Final training loss: 2.934
- Validation loss: 3.2140
- GPU utilization: 97.84%

#### Extended Training

- Additional 26 epochs
- Duration: 1 day, 19 hours, 38 minutes, 4 seconds
- Final validation loss: 3.12

### Fine-tuning Results

- Dataset: Orca-Math
- Duration: 4 days, 13 hours, 57 minutes
- Final metrics:
  - Training loss: 0.557
  - Validation loss: 0.5202

## Discussion

The research demonstrates the potential for creating compact, efficient models capable of mathematical reasoning. Key findings include:

1. The importance of model size and training data quality in achieving good performance.
2. The effectiveness of knowledge distillation from larger models like GPT-4 Turbo.
3. The challenge of balancing model size with computational constraints.
4. The discrepancy between validation loss and generated output coherence, highlighting the need for comprehensive evaluation methods.

## Conclusion

Math Nano GPT represents a significant step towards creating accessible, efficient language models for mathematical reasoning. While challenges remain, the research provides valuable insights into model compression, knowledge distillation, and the development of specialized language models.

## Future Work

1. Exploration of more advanced compression techniques
2. Investigation of alternative attention mechanisms
3. Development of improved evaluation methodologies
4. Few-shot benchmark development
5. Exploration of different types of attention mechanisms and architecture variants

## Installation and Usage

### Requirements

```
torch==1.9.0
transformers==4.11.3
datasets==1.11.0
wandb==0.12.1
boto3==1.18.44
tqdm==4.62.3
numpy==1.21.2
scikit-learn==0.24.2
matplotlib==3.4.3
tensorboard==2.6.0
pandas==1.3.3
scipy==1.7.1
nltk==3.6.3
regex==2021.8.28
```

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/math-nano-gpt.git
   cd math-nano-gpt
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

- Pre-training:
  ```
  python train_nano.py
  ```

- Fine-tuning:
  ```
  python finetuning.py
  ```

- Generate text:
  ```
  python generate.py
  ```

- Interactive Q&A:
  ```
  python askSLM.py
  ```

## Project Structure

- `main_model.py`: Core Math Nano GPT architecture
- `train_nano.py`: Pre-training script
- `finetuning.py`: Fine-tuning script
- `more_training.py`: Extended training script
- `generate.py`: Text generation script
- `askSLM.py`: Interactive Q&A script
- `token_train.py`, `token_val.py`: Data tokenization scripts
- `test.py`: Tokenized dataset testing script

## Acknowledgments

I extend my deepest gratitude to:

- Dr. Kaća Bradonjić (Hampshire College)
- Dr. Jaime Davila (UMASS Amherst)
- Dr. Ethan Ludwin-Peery (Hampshire College)
- Elliot Arledge

## References

[Include full list of references from your thesis]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Rupert Tawiah-Quashie - [rupertquashie@gmail.com]
```
