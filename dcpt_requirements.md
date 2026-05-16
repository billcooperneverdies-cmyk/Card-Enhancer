# DCPT Pipeline Requirements & Architecture

## Core Components
- **Valuation**: Predicting market value of cards.
- **Defect Classification**: Identifying and categorizing degradation types.
- **Node-Specific Attention Mechanism**:
    - Unique projection matrices: $W_Q[i], W_K[i], W_V[i], W_O[i], W_{ff}[i]$ for each node.
    - Attention coefficients adjusted by sigmoid of topological descriptors ($T_j^*$).
    - Formula: $\alpha_{ij} \propto \text{softmax}_j(\text{LeakyReLU}(W_{Q,i} h_i \cdot W_{K,i} h_j)) \cdot \sigma(5 \cdot T_j^*)$.

## Data Structure
- **Node Features (h)**: `[num_samples, num_nodes, in_features]`.
- **Adjacency Matrix (adj)**: `[num_samples, num_nodes, num_nodes]`.
- **Topological Descriptors (T_star)**: `[num_samples, num_nodes]`.
- **Prices**: `[num_samples]` (for valuation).
- **Labels**: `[num_samples]` (for defect classification).

## Modules to Implement
1. `models/dcpt_model.py`: Core DCPT model.
2. `training/trainers.py`: Multi-task training logic.
3. `tda_optimization/tda_core.py`: `TDACacheManager`, `TDAApproximator`, and `TDAGPUAccelerator` (placeholder).
4. `utils/data_utils.py`: `CardDataset` and synthetic data generation.
5. `tests/test_pipeline.py`: Comprehensive test suite.
6. `run_pipeline.py`: End-to-end execution script.

## Configuration Defaults
- `num_nodes = 10`
- `in_features = 64`
- `hidden_features = 128`
- `num_classes = 6`
- `batch_size = 32`
- `epochs = 20`
