import torch 
import torch.nn as nn 

class BigModel(nn.Module):
    def __init__(self, vector_size, scale=3):
        super().__init__()

        self.linear1 = nn.Linear(vector_size, scale * vector_size)
        self.relu1 = nn.ReLU() 

        self.linear2 = nn.Linear(scale * vector_size, scale * vector_size)
        self.relu2 = nn.ReLU() 

        self.linear3 = nn.Linear(scale * vector_size, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.relu2(x)

        # letting the model figure out final scaling
        out = self.linear3(x)
        return out
