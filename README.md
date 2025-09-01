# Handwritten Digit Recognition with MNIST dataset (no ML framework)

This project implements a fully connected deep neural network from scratch in NumPy to recognize handwritten digits (0–9) from the MNIST dataset.

It includes:

- Forward & backward propagation
- Softmax output with categorical cross-entropy loss
- Mini-batch gradient descent
- Adam optimizer
- Functions for training, prediction, and error analysis

The goal is to provide a learning-friendly, from-scratch implementation of a multi-layer neural network without relying on deep learning frameworks like TensorFlow or PyTorch.

## Features

- He initialization for better convergence
- ReLU activations for hidden layers
- Softmax activation for multiclass classification
- Categorical cross-entropy loss
- Mini-batch training
- Adam optimizer (with bias correction)
- Accuracy evaluation and misclassification inspection

## Project Structure

```
│── model.ipynb # Jupyter Notebook with training & experiments
│── mnist_dataloader.py # Functions to load the MNIST dataset
│── model_functions.py # Functions used in the model implementation
│── mathematical_foundations.pdf # Mathematical explanation of the implemented functions
│── README.md # Project documentation
```

## Requirements

- Python 3.8+
- NumPy
- matplotlib # for examples visualization
- An environment to visualize and run a Jupyter Notebook file

## Training the Model

1. Load and preprocess MNIST (flatten images to shape (784, m) and one-hot encode labels).

2. Define network architecture and hyperparameters. Example:

```python
layers_dims = [784, 128, 64, 10] # 2 hidden layers
```

3. Train the model:

```python
parameters, costs = L_layer_model(
X_train, Y_train, layers_dims,
learning_rate=0.001,
num_epochs=1000,
mini_batch_size=128,
print_cost=True
)
```

4. Evaluate:

```python
preds = predict(X_test, Y_test, parameters)
```

## Results

The implementation achieves 98% accuracy on a 10,000 samples test set.

You can also inspect misclassified digits:

```python
show_wrongly_classified(X_test, Y_test, parameters, num_to_show=10)
```

## Acknowledgments

This implementation was inspired by the [Deep Learning Specialization](https://www.deeplearning.ai/) course by Andrew Ng (DeepLearning.AI).

## License

This project is released under the MIT License. Feel free to use, modify, and share.
