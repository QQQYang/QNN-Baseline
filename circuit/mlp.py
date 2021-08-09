#encoding=utf-8
"""
Implementation of multi-layer perceptron
--------------------------------------
Author: xxx
Email: xxx@xxx.com
"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.model = []
        for i in range(len(dim)-1):
            self.model.append(nn.Linear(dim[i], dim[i+1], bias=True))
            self.model.append(nn.ReLU())
        # self.model.append(nn.Dropout())
        self.model.append(nn.Linear(dim[-1], 2, bias=True))
        self.model = nn.Sequential(*self.model)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, x):
        return self.model(x)