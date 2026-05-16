# DCPT Training Pipeline

## Overview
The **Degradation Classification Pre-Training (DCPT)** pipeline is a high-performance framework designed for **valuation** and **defect classification** of collectible cards. It integrates a novel graph neural network architecture with **node-specific attention** and **topological data analysis (TDA)**.

## Key Features
- **Node-Specific Attention**: Employs unique projection matrices ($W_Q, W_K, W_V, W_O$) for each node in the graph, allowing for granular feature extraction.
- **TDA-Modulated Attention**: Attention coefficients are dynamically adjusted using a sigmoid function of topological descriptors ($T_j^*$), prioritizing structurally significant regions.
- **Multi-Task Learning**: Jointly optimizes for market value prediction (regression) and defect identification (classification).
- **TDA Optimization**: Includes caching and approximation modules to handle computationally intensive topological calculations efficiently.

## Project Structure
- `dcpt_pipeline/models/dcpt_model.py`: Core architecture implementation.
- `dcpt_pipeline/training/trainers.py`: Multi-task training logic and metrics tracking.
- `dcpt_pipeline/tda_optimization/tda_core.py`: Caching, approximation, and GPU acceleration placeholders.
- `dcpt_pipeline/utils/data_utils.py`: Data loading and synthetic data generation.
- `dcpt_pipeline/tests/test_pipeline.py`: Comprehensive test suite for scientific validation.
- `run_pipeline.py`: End-to-end execution script.

## Getting Started
1. **Install Dependencies**:
   ```bash
   pip install torch torchvision torchaudio loguru scikit-learn joblib opencv-python pandas numpy tqdm
   ```
2. **Run Tests**:
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python3 dcpt_pipeline/tests/test_pipeline.py
   ```
3. **Execute Pipeline**:
   ```bash
   python3 run_pipeline.py
   ```

## Scientific Principles
The pipeline follows first principles by grounding its attention mechanism in the topological complexity of the input data. The inclusion of the **BPDA Gate** logic in tests ensures that the system can handle non-differentiable boundaries during adversarial training or complex defense scenarios by approximating gradients during the backward pass.
