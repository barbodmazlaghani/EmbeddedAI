Project Overview

This project focuses on implementing and optimizing neural networks on embedded systems with limited resources. The primary goals are to:

Implement neural network models using Python and SystemVerilog.
Optimize these models for deployment on hardware with constrained computational resources.
Key Components
Neural Networks:

The project utilizes a Multi-Layer Perceptron (MLP) for tasks such as image processing and edge detection.
MLP networks are chosen due to their lightweight nature and low computational requirements, making them suitable for embedded systems.
Autoencoder Design:

An autoencoder, composed of an encoder and decoder, is implemented for data compression tasks.
The project employs datasets such as MNIST and NotMNIST for training and testing the autoencoder.
Quantization Techniques:

To reduce model size and computational load, quantization techniques are applied. This involves converting weights from floating-point to fixed-point representations, significantly decreasing memory usage and model size without substantial loss of accuracy.

Project Steps:

Step 1: Python scripts (hw-mnist.ipynb and hw-notmnist.ipynb) are used to train the autoencoder on the MNIST and NotMNIST datasets, apply quantization, and save the quantized weights for hardware deployment.
Step 2: Using SystemVerilog, the quantized models are implemented in hardware. The code involves modules for operations like MAC (Multiply and Accumulate) and activation functions such as ReLU.
Hardware Implementation:

The project includes the design and simulation of neural network layers in SystemVerilog, considering parameters such as input width, output width, and the number of neurons per layer.
A testbench (testbench.sv) is created to validate the hardware implementation against the software model.
Required Software
Python for training neural networks and applying quantization.
ModelSim or a similar HDL simulator for testing SystemVerilog implementations.
How to Use
Run the provided Python scripts to train the neural network and save the quantized model.
Use the SystemVerilog files to simulate the model on a hardware description language simulator.
Follow the project instructions to fine-tune and optimize the model for your specific embedded hardware platform.