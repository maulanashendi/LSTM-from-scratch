import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L

class LSTMbyHand(L.LightningModule):
    def __init__(self):
        super().__init__()
        
        # Initialize mean and standard deviation for normal distribution
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        # Initialize LSTM gate parameters (weights and biases)
        self.w_blue1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.w_blue2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.b_blue1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.w_green1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.w_green2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.b_green1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.w_orange1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.w_orange2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.b_orange1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.w_purple1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.w_purple2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.b_purple1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
    
    
    def lstm_unit(self, input_value, long_memory, short_memory):

        # Calculate the "forget" gate for long-term memory
        long_remember_percent = torch.sigmoid(
            (short_memory * self.w_blue1) + (input_value * self.w_blue2) + self.b_blue1
        )
        
        # Calculate the "input" gate for updating long-term memory
        potential_remember_percent = torch.sigmoid(
            (short_memory * self.w_green1) + (input_value * self.w_green2) + self.b_green1
        )
        
        # Calculate the candidate memory content to be added to long-term memory
        potential_remember = torch.tanh(
            (short_memory * self.w_orange1) + (input_value * self.w_orange2) + self.b_orange1
        )

        # Update long-term memory using the forget and input gates
        updated_long_memory = (
            (long_memory * long_remember_percent) + 
            (potential_remember_percent * potential_remember)
        )
        
        # Calculate the "output" gate for short-term memory
        output_percent = torch.sigmoid(
            (short_memory * self.w_purple1) + (short_memory * self.w_purple2) + self.b_purple1
        )
        
        # Update short-term memory based on the updated long-term memory
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent

        return [updated_long_memory, updated_short_memory]
    
    
    def forward(self, input):
        long_memory = 0
        short_memory = 0
        
        # Iterate through the input sequence
        for i in range(input.size(0)):
            input_value = input[i]  # Get the current input value
            long_memory, short_memory = self.lstm_unit(input_value, long_memory, short_memory)

        return short_memory
    
    # Configure the optimizer for the training process
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
    
    
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch  
        output_i = self.forward(input_i[0])  
        loss = (output_i - label_i) ** 2  # Calculate the loss (MSE)

        self.log("train_loss", loss)

        if label_i == 0:
            self.log('out_0', output_i)
        else:
            self.log('out_1', output_i)

        return loss
