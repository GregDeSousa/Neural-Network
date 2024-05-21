# Neural Network Visualizer

This project is a university project from the University of Pretoria which contains a neural network implementation and a visualizer for the network using Tkinter. The neural network is trained using backpropagation on a dataset read from a CSV file. The visualizer helps to display the network's structure, activations, and weights.

## Description

The project includes:
- A neural network implementation with forward propagation and backpropagation.
- A visualizer to display the network's structure, activations, and weights using Tkinter.

## Files

- `neural_network.py`: Main code for the neural network and visualizer.

## Usage

To use the neural network and visualizer, run the `neural_network.py` script. The script includes test inputs and will display the network's structure, activations, and weights. The network is structured using 12 binary inputs and 3 binary outputs using binary regression, which, for the purpose of the practical, represent the prior history of moves played by the bot and neural network, and the output (Rock, Paper, Scissors). However, the network can be repurposed to meet the criteria of 12 inputs and 3 outputs. The hyperparameters can be adjusted at the top of the script to better suit the needs of the model it is trained on. The visualizer will first show the untrained model in a Tkinter window, then it will proceed to train the model on the given data file. Once completed, the script will successively show 3 Tkinter windows, displaying the outputs of the neural network for the first 3 data items in the dataset to visually verify the model efficacy. In the default configuration of the code, it will show the correct counter move to the second column in the `data1.csv` file where the first column is the input of historical moves that the bots and the network played. The top output neuron is rock, the middle output neuron is paper, and the bottom output neuron is scissors.

## Installation

To run the code, you need to have Python and Tkinter installed. You can install the required packages using pip:

```bash
pip install numpy
pip install tkinter
