Project Overview

This project is focused on implementing and synthesizing neural networks on embedded systems with limited resources. The primary objectives are to:

Develop and optimize a neural network model using Python and Verilog.
Synthesize the model for hardware using synthesis tools.
Key Components
Neural Network Implementation:

The project utilizes a Multi-Layer Perceptron (MLP) for classification tasks.
MLP networks are chosen due to their efficiency in managing a wide range of tasks, including classification and regression, and their suitability for deployment on hardware with limited resources.
Dataset:

The NotMNIST dataset, consisting of alphabet letters A, B, and C, is used for training the neural network. The dataset is pre-processed for training, requiring no further normalization.
Quantization Techniques:

Quantization is employed to convert continuous values into discrete levels, significantly reducing model size and computational requirements. This technique is especially useful for neural networks deployed on hardware platforms with limited memory and processing capabilities.

Project Steps:

Step 1: Develop the classification model using Python (hw-notmnist.ipynb). Train the model to achieve a minimum accuracy of 96% on the NotMNIST dataset. Record metrics such as precision, recall, and F1-score.
Step 2: Implement the neural network in Verilog and synthesize the model using synthesis tools like Yosys. The synthesis involves converting the Python-trained model to hardware description language (HDL) code that can be deployed on FPGA or ASIC platforms.
Hardware Synthesis:

The project includes the design of neural network components in Verilog, such as MAC (Multiply and Accumulate) units and activation functions (ReLU and Clip modules).
A testbench (testbench.sv) is created to validate the Verilog implementation against the Python model.
Optimization and Synthesis:

Synthesis tools like Yosys are used to optimize the Verilog code for hardware implementation. This involves minimizing the number of logic cells and optimizing the use of resources such as LUTs (Look-Up Tables) and DFFs (D Flip-Flops).
Required Software
Python for neural network development and initial training.
ModelSim or a similar HDL simulator for testing Verilog implementations.
Yosys for synthesis of Verilog code to hardware.
How to Use
Run the provided Python script to train the neural network on the NotMNIST dataset and save the model parameters.
Use the Verilog files to implement and simulate the neural network on an HDL simulator.
Synthesize the Verilog code using Yosys to generate the hardware implementation for FPGA or ASIC.