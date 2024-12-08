import torch.nn as nn
import torch.nn.functional as F
import torch

class Causal_Encoder_1(nn.Module): 
    # LE net
    # q_theta_1(d|x)
    def __init__(self, input_dim, output_dim):
        super(Causal_Encoder_1, self).__init__()
        self.output_dim = output_dim
        self.coding = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, self.output_dim))
        
    def forward(self, x):
        d = self.coding(x)
        return d
    
class Causal_Encoder_2(nn.Module):

    # q_theta_2(z|d,x)

    def __init__(self, input_dim, output_dim):
        super(Causal_Encoder_2, self).__init__()
        self.h_dim = output_dim
        self.coding = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, self.h_dim*2))
        
    def forward(self, x):
        statistics = self.coding(x)
        mu = statistics[:,:self.h_dim]
        std = F.softplus(statistics[:,self.h_dim:])
        return mu, std
    
class Causal_Decoder_1(nn.Module):
    # p_omega_1(l|d,x)
    def __init__(self, input_dim, output_dim):
        super(Causal_Decoder_1, self).__init__()
        self.output_dim = output_dim
        self.coding = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, output_dim))
    def forward(self, x):
        d = self.coding(x)
        return d

class Causal_Decoder_2(nn.Module):

    # p_omega_1(x|d,z)
    def __init__(self, input_dim, output_dim):
        super(Causal_Decoder_2, self).__init__()
        self.output_dim = output_dim
        self.coding = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, output_dim))
    def forward(self, x):
        d = self.coding(x)
        return d

