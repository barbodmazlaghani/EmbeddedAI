Project Overview

This project focuses on implementing and synthesizing neural networks on embedded systems with limited resources. The primary objectives are to:

Develop and optimize a convolutional neural network (CNN) model using Python and HLS4ML.
Synthesize the model for hardware implementation using Vivado and HLS tools.
Key Components
Neural Network Implementation:

The project utilizes a Convolutional Neural Network (CNN) for image classification tasks.
CNNs are chosen due to their effectiveness in handling image and video data, utilizing convolutional layers, pooling layers, and fully connected layers to extract and classify features.
Dataset:

The Street View House Numbers (SVHN) dataset is used for training the CNN model. The dataset consists of images of house numbers which are used to train and evaluate the model's classification accuracy.
Quantization and Pruning Techniques:

Quantization: This process reduces the number of bits used to represent the network parameters, decreasing the model size and computational load.
Pruning: This technique reduces the number of parameters in the model by removing weights that contribute less to the output, allowing for a more efficient model with minimal loss of accuracy.

Project Steps:

Step 1: Train the CNN model on the SVHN dataset using Python (hw-3.ipynb). Quantize the model and prune it to achieve maximum sparsity without exceeding a 1% loss in accuracy.
Step 2: Convert the trained model into a hardware-suitable format using HLS4ML. Synthesize the model for hardware deployment using Vivado.
Step 3: Perform performance evaluations on the quantized and pruned models, comparing metrics such as accuracy, latency, and resource usage.
Hardware Synthesis:

Use the Vivado tool for synthesizing the model into a hardware description. The synthesis involves converting the high-level Python model into an HDL (Hardware Description Language) format, optimized for FPGA implementation.
Validate the synthesized model using Vivado’s simulation and synthesis reports to ensure the model meets the desired performance criteria.
Optimization and Analysis:

Evaluate the models using various metrics such as accuracy with bit-accurate emulation, latency, and resource usage (LUTs, DFFs, etc.).
Compare different synthesis results to select the optimal model configuration based on accuracy, hardware resource efficiency, and latency.
Required Software
Python for neural network development and initial training.
Google Colab for running Python scripts and initial training if local resources are limited.
Vivado for synthesizing the neural network model into hardware-compatible code.
HLS4ML for converting high-level neural network models to FPGA-optimized HDL code.
How to Use
Run the provided Python script to train, quantize, and prune the neural network on the SVHN dataset.
Use HLS4ML to convert the trained model into a format suitable for FPGA synthesis.
Synthesize the HDL code using Vivado and validate the model performance.
Compare different models based on synthesis reports and select the best configuration for deployment.