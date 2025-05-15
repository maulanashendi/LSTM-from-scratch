# Custom LSTM Implementation Using PyTorch Lightning

This project contains a manual implementation of the Long Short-Term Memory (LSTM) unit using PyTorch and PyTorch Lightning. Rather than relying on built-in modules such as `nn.LSTM`, this implementation builds all components of an LSTM from scratch. The goal is to provide a deeper understanding of how LSTM works internally—mathematically and structurally.

## Overview

LSTM is a variant of Recurrent Neural Networks (RNN) designed to mitigate the vanishing gradient problem in long sequential data. The key feature of LSTM is its ability to retain information over long sequences using two memory components:

* **Long-Term Memory** (cell state)
* **Short-Term Memory** (hidden state)

These memory states are modulated using three types of gates:

* **Forget Gate**: controls what to discard from long-term memory
* **Input Gate**: controls what new information to add
* **Output Gate**: controls what part of memory to output

## Architecture and Mathematical Formulation

This project implements all gates using `nn.Parameter` to manually control the weights and biases for each component.

### 1. Forget Gate

The forget gate determines how much of the previous long-term memory should be retained:

$$
f_t = \sigma(W_{f,x} \cdot x_t + W_{f,h} \cdot h_{t-1} + b_f)
$$

In the code:

```python
long_remember_percent = torch.sigmoid(
    (short_memory * self.w_blue1) + (input_value * self.w_blue2) + self.b_blue1
)
```

### 2. Input Gate and Candidate Memory

The input gate determines how much new information should be written to the memory:

$$
i_t = \sigma(W_{i,x} \cdot x_t + W_{i,h} \cdot h_{t-1} + b_i)
$$

$$
\tilde{C}_t = \tanh(W_{c,x} \cdot x_t + W_{c,h} \cdot h_{t-1} + b_c)
$$

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

In the code:

```python
potential_remember_percent = torch.sigmoid(...)
potential_remember = torch.tanh(...)
updated_long_memory = (long_memory * long_remember_percent) + (potential_remember_percent * potential_remember)
```

### 3. Output Gate

The output gate determines how much of the updated memory should be sent to the output:

$$
o_t = \sigma(W_{o,x} \cdot x_t + W_{o,h} \cdot h_{t-1} + b_o)
$$

$$
h_t = o_t \odot \tanh(C_t)
$$

In the code:

```python
output_percent = torch.sigmoid(...)
updated_short_memory = torch.tanh(updated_long_memory) * output_percent
```

## Code Structure

* `LSTM.py`: Contains the `LSTMbyHand` class with gate-level manual parameter initialization and memory update logic.
* `main.ipynb`: Demonstrates training and evaluating the model in an interactive Jupyter Notebook environment.

This implementation uses PyTorch Lightning’s `LightningModule` structure to simplify training and logging logic.

## How to Run

1. Install dependencies:

```bash
pip install torch lightning
```

2. Open and run `main.ipynb` to see how the model is trained and tested.

## Notes

* This project is designed for educational and experimental purposes.
* All LSTM gates and memory mechanisms are built manually using basic PyTorch components.
* Suitable for learning, visualization, and research purposes.
