import pandas as pd
import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from torchvision import transforms

# defining AE model
class AE(nn.Module):
    # x--> fc1--> tanh--> fc2--> z--> fc3--> tanh--> fc4--> x'
    
    def __init__(self, shape, h_dim, z_dim):
        #call the super constructor
        super(AE, self).__init__()
        self.shape = shape
        
        self.fc1 = nn.Linear(shape[1]*shape[2], h_dim) #encode
        self.fc2 = nn.Linear(h_dim, z_dim) #encode
        
        self.fc3 = nn.Linear(z_dim, h_dim) #decode
        self.fc4 = nn.Linear(h_dim, shape[1]*shape[2]) #decode
        
        #activation function
        self.tanh = nn.Tanh()
        
        #initializing weights from xavier normal distribution
        nn.init.xavier_normal(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_normal(self.fc2.weight, gain=np.sqrt(2))
        nn.init.xavier_normal(self.fc3.weight, gain=np.sqrt(2))
        nn.init.xavier_normal(self.fc4.weight, gain=np.sqrt(2))
        
        def encode(self, x):
            # x--> fc1--> tanh--> fc2--> z
            h = self.tanh(self.fc1(x))
            z = self.fc2(h)
            return z
        
        def decode(self, z, x):
            # z --> fc3 --> tanh --> fc4 --> x'
            h = self.tanh(self.fc3(z))
            reconstructed_x = self.fc4(h)
            return reconstructed_x.view(x.size())
        
        def forward(self, x):
            #flatten the input and pass it to encode
            #one row (sample) many columns (features) at a time
            #-1 calculates the number of rows
            # shape(1200, 14, 15) means 1200 samples 14*15=210 features
            z = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
            return self.decode(z, x)
        
#defining the AE model with dropout
class AE_dropout(nn.Module):
    # x--> dropout--> fc1--> tanh--> dropout--> fc2--> z--> dropout--> fc3--> tanh--> dropout--> fc4--> x'
    def __init__(self, shape, h_dim, z_dim):
        super(AE_dropout, self).__init__()
        self.shape = shape
        
        self.fc1 = nn.Linear(shape[1]*shape[2], h_dim) #encode
        self.fc2 = nn.Linear(h_dim, z_dim) #encode
        
        self.fc3 = nn.Linear(z_dim, h_dim) #decode
        self.fc4 = nn.Linear(h_dim, shape[1]*shape[2]) #decode

        self.dout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        
        #initializing weights
        nn.init.xavier_normal(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_normal(self.fc2.weight, gain=np.sqrt(2))
        nn.init.xavier_normal(self.fc3.weight, gain=np.sqrt(2))
        nn.init.xavier_normal(self.fc4.weight, gain=np.sqrt(2))

    def encode(self, x):
        # x--> dropout--> fc1--> tanh--> dropout--> fc2--> z
        dx = self.dout(x)
        h = self.tanh(self.fc1(dx))
        h = self.dout(h)
        z = self.fc2(h)
        return z

    def decode(self, z, x):
        # z--> dropout--> fc3--> tanh--> dropout--> fc4--> x'
        dz = self.dout(z)
        h = self.tanh(self.fc3(dz))
        h = self.dout(h)
        recon_x = self.fc4(h)
        return recon_x.view(x.size())
    
    def forward(self, x):
        #flatten the input and pass it to encode
        z = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        return self.decode(z, x)
        
#defining VAE model
class VAE(nn.Module):
    # x--> fc1--> relu--> fc2--> z--> fc3--> relu--fc4--> x'
    def __init__(self, shape, layer1, layer2, isCuda):
        '''
        input size: (batch, sequence_length, feature)
        shape: tuple (shape[1], shape[2]) / (sequence_length, feature) of c_train
        layer1: dimension of hidden layer1
        layer2: dim of hidden layer2
        '''
        super(VAE, self).__init__()

        self.shape = shape
        self.isCuda = isCuda
        
        self.fc1 = nn.Linear(shape[1]*shape[2], layer1) 
        #this is self.fc_mean
        self.fc21 = nn.Linear(layer1, layer2) #encode
        #this is self.fc_var
        self.fc22 = nn.Linear(layer1, layer2) #encode
        self.fc3 = nn.Linear(layer2, layer1) #decode
        self.fc4 = nn.Linear(layer1, shape[1]*shape[2]) #decode
    
        self.relu = nn.ReLU()
        
        if self.isCuda:
            self.cuda()

        #initialize weights
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc21.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc22.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc3.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc4.weight, gain=np.sqrt(2))

    def encode(self, x):
        #x--> fc1--> relu--> fc21
        #x--> fc1 --> relu --> fc22
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar/2)
        #Torch.normal_() : fills self tensor with elements samples from the normal distribution parameterized by mean and std.
        if self.isCuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps*std + mu

    def decode(self, z, x):
        #z --> fc3 --> relu --> fc4
        h3 = self.relu(self.fc3(z))
        return self.fc4(h3).view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        #mu= mean
        #logvar = log variance
        mu, logvar = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        z = self.reparametrize(mu, logvar)
        return self.decode(z, x), mu, logvar
    
class VAE_dropout(nn.Module):
    # Applying dropout to both input and hidden layers
    # x--> dropout--> fc1--> relu--> dropout--> fc2--> z--> dropout--> fc3 --> relu--> dropout--> fc4--> x'
    def __init__(self, shape, layer1, layer2, isCuda):
        '''
        input size: (batch, sequence_length, feature)
        shape: tuple (shape[1], shape[2])/(sequence_length, feature) of c_train
        layer1: dimension of hidden layer1
        layer2: dim of hidden layer2
        '''
        super(VAE_dropout, self).__init__()

        self.shape = shape
        self.isCuda = isCuda
        
        self.fc1 = nn.Linear(shape[1]*shape[2], layer1) #encode
        self.fc21 = nn.Linear(layer1, layer2) #encode
        self.fc22 = nn.Linear(layer1, layer2) #encode

        self.fc3 = nn.Linear(layer2, layer1) #decode
        self.fc4 = nn.Linear(layer1, shape[1]*shape[2]) #decode
    
        self.relu = nn.ReLU()
        self.dout = nn.Dropout(p=0.2)
        
        if self.isCuda:
            self.cuda()

        #initialize weights
        nn.init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc21.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc22.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc3.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.fc4.weight, gain=np.sqrt(2))

    def encode(self, x):
        # x--> dropout--> fc1--> relu--> dropout--> fc2
        # x--> dropout--> fc1-> relu--> dropout--> fc21
        # x--> dropout--> fc1--> relu--> dropout--> fc22
        dx = self.dout(x)
        h1 = self.relu(self.fc1(dx))
        h1 = self.dout(h1)
        return self.fc21(h1), self.fc22(h1)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar/2)
        if self.isCuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps*std + mu

    def decode(self, z, x):
        # z --> dropout --> fc3 --> relu --> dropout -->fc4 --> x'
        dz = self.dout(z)
        h3 = self.relu(self.fc3(dz))
        h3 = self.dout(h3)
        return self.fc4(h3).view(x.size())
    
    def forward(self, x):
        #flatten input and pass to encode
        #mu= mean
        #logvar = log variational
        mu, logvar = self.encode(x.view(-1, self.shape[1]*self.shape[2]))
        z = self.reparametrize(mu, logvar)
        return self.decode(z, x), mu, logvar

#defining GRU based AE
class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)#batch first=true means batch is first dimension
        #x shape is batch_size, seq, input_size
        
        #initializing the weights from a xavier normal distribution
        nn.init.xavier_uniform(self.gru.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.gru.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, input):
        tt = torch.cuda if self.isCuda else torch
        #initial hidden layer
        h0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size).zero_(), requires_grad=False)
        encoded_input, hidden = self.gru(input, h0)
        return encoded_input, hidden

      
class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
        #initializing the weights
        nn.init.xavier_uniform(self.gru.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.gru.weight_hh_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.linear.weight, gain=np.sqrt(2))
        
    def forward(self, encoded_input, hidden):
        tt = torch.cuda if self.isCuda else torch
        decoded_output, _ = self.gru(encoded_input, hidden)
        decoded_output = self.linear(decoded_output) 
        return decoded_output

class GRUAE(nn.Module):
    # x--> lstm--> z--> lstm--> fc--> x'
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(GRUAE, self).__init__()
        self.encoder = EncoderGRU(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderGRU(hidden_size, input_size, num_layers, isCuda)
        
    def forward(self, input):
        encoded_input, hidden = self.encoder(input)
        decoded_output = self.decoder(encoded_input, hidden)
        return decoded_output
        
        
        
        
        