# Deep Learning from Scratch - Implementations

This directory contains pure-Python (NumPy-based) implementations of deep learning foundations, inspired by "Deep Learning from Scratch".

## Source Files Guide
- **`2_perceptron.ipynb` - `6_optimization.ipynb`**: Progressive implementation tutorials.
- **`functions.py`**: Common activation (Sigmoid, ReLU, Softmax) and loss functions (Cross Entropy).
- **`layers.py`**: Object-oriented layers (Affine, ReLU, Softmax-with-Loss) with `forward()` and `backward()` methods.
- **`gradient.py`**: Numerical gradient calculation.
- **`two_layer_net4.py`**: A complete implementation of a 2-layer neural network.
- **`train_neuralnet.py`**: Training loop that integrates data loading, loss monitoring, and weight updates.

## Implementation Guidelines
1.  **NumPy First**: Do NOT use higher-level deep learning frameworks (like PyTorch or TensorFlow) unless explicitly requested.
2.  **Layer-wise Modularity**: All new layers MUST follow the `forward()` and `backward()` interface in `layers.py`.
3.  **Vectorization**: Always prefer NumPy's vectorized operations (`np.dot`, `np.sum`) over explicit Python loops for efficiency.
4.  **Numerical Validation**: When implementing backpropagation, use `numerical_gradient` from `gradient.py` to verify analytical gradients.

## Usage
- Run Jupyter Notebooks for interactive learning.
- Execute `train_neuralnet.py` to see the complete training process on the MNIST dataset.
