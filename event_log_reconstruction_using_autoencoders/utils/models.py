import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
  
      
class AE(nn.Module):
    # x --> fc1 --> fc2 --> sigmoid --> z --> fc3 --> fc4 --> sigmoid --> x'
    def __init__(self, shape, h_dim, z_dim):
        super(AE, self).__init__()
        self.shape = shape
        
        self.fc1 = nn.Linear(shape[1]*shape[2], h_dim) #encode
        self.fc2 = nn.Linear(h_dim, z_dim) #decode
        
        self.fc3 = nn.Linear(z_dim, h_dim) #decode
        self.fc4 = nn.Linear(h_dim, shape[1]*shape[2]) #decode
    
        #self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        #initialize weights
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))

    def encode(self, x):
        # x --> fc1 --> fc2 --> sigmoid --> z
        h = self.fc1(x)
        z = self.sigmoid(self.fc2(h))
        return z

    def decode(self, z, x):
        # z --> fc3 --> fc4 --> sigmoid --> x'
        h = self.fc3(z)
        recon_x = self.sigmoid(self.fc4(h))
        return recon_x.view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        z = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        return self.decode(z, x)
    
class AE_tanh(nn.Module):
    # x --> fc1 --> tanh --> fc2 --> tanh --> z --> fc3 --> tanh --> fc4 --> sigmoid --> x'
    def __init__(self, shape, h_dim, z_dim):
        super(AE_tanh, self).__init__()
        self.shape = shape
        
        # X --> fc1 --> Z --> fc2 --> X'
        self.fc1 = nn.Linear(shape[1]*shape[2], h_dim) #encode
        self.fc2 = nn.Linear(h_dim, z_dim) #decode
        
        self.fc3 = nn.Linear(z_dim, h_dim) #decode
        self.fc4 = nn.Linear(h_dim, shape[1]*shape[2]) #decode
    
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        #initialize weights
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))

    def encode(self, x):
        # x --> fc1 --> tanh --> fc2 --> tanh --> z
        h = self.tanh(self.fc1(x))
        z = self.tanh(self.fc2(h))
        return z

    def decode(self, z, x):
        # z --> fc3 --> tanh --> fc4 --> sigmoid --> x'
        h = self.tanh(self.fc3(z))
        recon_x = self.sigmoid(self.fc4(h))
        return recon_x.view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        z = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        return self.decode(z, x)
    
class AE_dropout(nn.Module):
    # x --> dropout --> fc1 --> tanh ---> dropout --> fc2 --> z --> dropout --> fc3 --> tanh --> dropout --> fc4 --> sigmoid --> x'
    def __init__(self, shape, h_dim, z_dim):
        super(AE_dropout, self).__init__()
        self.shape = shape
        
        self.fc1 = nn.Linear(shape[1]*shape[2], h_dim) #encode
        self.fc2 = nn.Linear(h_dim, z_dim) #encode
        
        self.fc3 = nn.Linear(z_dim, h_dim) #decode
        self.fc4 = nn.Linear(h_dim, shape[1]*shape[2]) #decode

        self.dout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        #initialize weights
        nn.init.xavier_normal(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_normal(self.fc2.weight, gain=np.sqrt(2))
        nn.init.xavier_normal(self.fc3.weight, gain=np.sqrt(2))
        nn.init.xavier_normal(self.fc4.weight, gain=np.sqrt(2))

    def encode(self, x):
        # x --> dropout --> fc1 --> tanh ---> dropout --> fc2 --> z
        dx = self.dout(x)
        h = self.tanh(self.fc1(dx))
        h = self.dout(h)
        z = self.fc2(h)
        return z

    def decode(self, z, x):
        # z --> dropout --> fc3 --> tanh --> dropout --> fc4 --> sigmoid --> x'
        dz = self.dout(z)
        h = self.tanh(self.fc3(dz))
        h = self.dout(h)
        recon_x = self.sigmoid(self.fc4(h))
        return recon_x.view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        z = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        return self.decode(z, x)


