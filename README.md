# TensorEase

TensorEase is a simple neural network library built from scratch using NumPy. It supports building and training custom deep learning models, including layers like linear (fully connected) and activation functions such as ReLU, LeakyReLU, and Sigmoid. TensorEase can be used to tackle both regression and classification tasks.

## Features

- Build custom neural networks using a `Sequential` model.
- Support for common layers like `Linear`.
- Various activation functions such as `ReLU` and `LeakyReLU`.
- Custom loss functions like Mean Squared Error (MSE).
- Gradient-based optimization through backpropagation.
- Capability to train and test models on real-world datasets like California Housing.
  
## Requirements

- Python 3.x
- NumPy
- Pandas (optional, for handling datasets)
- Scikit-learn (optional, for data preprocessing)

## Installation

1. Create a virtual environment:

   ```bash
   python3 -m venv venv

2. Activate the virtual environment (for macOS):
  ```bash
  source venv/bin/activate

3. Install the required libraries:
  pip install -r requirements.txt

## Usage

1. Load your dataset (synthetic or real-world) and preprocess it according to your task's requirements.
2. Construct a neural network model using the `Sequential()` class and define your layers with components such as `Linear` layers and activation functions like `ReLU`.
3. Train your model using the `.train()` method by specifying your training data and number of epochs.
4. Evaluate the model's performance using the `.predict()` method, comparing against your validation or test dataset.
5. Use built-in plotting methods or libraries like `matplotlib` to visualize model predictions versus actual values.

## Example Workflow

1. Load and preprocess your dataset (e.g., scaling features, splitting into training and test sets).
2. Define the model architecture:
   - Add layers, starting with an input layer that matches your dataset's features.
   - Include hidden layers with activation functions like ReLU or LeakyReLU.
   - End with an output layer suited for your regression or classification task.
3. Train the model using your training dataset.
4. Evaluate the performance with metrics such as Mean Squared Error (MSE) for regression tasks or accuracy for classification.
5. Visualize predictions versus actual results to assess model accuracy and performance.

## Roadmap

- Implement advanced optimizers (e.g., Adam, RMSprop).
- Add more activation functions and regularization techniques like dropout.
- Support for various data augmentation and preprocessing utilities.
- Introduce user-friendly APIs for model evaluation and visualization.