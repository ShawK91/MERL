import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch
import torch.nn.functional as F
import numpy as np, sys
import matplotlib.pyplot as plt
from scipy.special import expit
import random, fastrand, math
import pickle as cPickle
import copy


#MMU Bundle
class MMU(nn.Module):
    def __init__(self, input_dim, hid_dim, mem_dim, out_dim):
        super(MMU, self).__init__()

        self.input_dim = input_dim; self.hid_dim = hid_dim; self.mem_dim = mem_dim; self.out_dim= out_dim

        # Input gate
        self.w_inpgate = nn.Linear(input_dim, hid_dim)
        self.w_rec_inpgate = nn.Linear(out_dim, hid_dim)
        self.w_mem_inpgate = nn.Linear(mem_dim, hid_dim)

        # Block Input
        self.w_inp = nn.Linear(input_dim, hid_dim)
        self.w_rec_inp = nn.Linear(out_dim, hid_dim)

        # Read Gate
        self.w_readgate = nn.Linear(input_dim, mem_dim)
        self.w_rec_readgate = nn.Linear(out_dim, mem_dim)
        self.w_mem_readgate = nn.Linear(mem_dim, mem_dim)


        # Memory Decoder
        self.w_decoder = nn.Linear(hid_dim, mem_dim)

        # Write Gate
        self.w_writegate = nn.Linear(input_dim, mem_dim)
        self.w_rec_writegate = nn.Linear(out_dim, mem_dim)
        self.w_mem_writegate = nn.Linear(mem_dim, mem_dim)

        # Memory Encoder
        self.w_encoder = nn.Linear(mem_dim, hid_dim)

        #Adaptive components
        self.mem = None
        self.out = None

        #Output weights
        self.w_hid_out = Parameter(torch.rand(out_dim, mem_dim), requires_grad=True)

        # History for RRN
        self.hist_steps = 5
        self.rnn_history = []#np.zeros([5,20,20])

    def reset(self, batch_size):
        # Adaptive components
        self.mem = Variable(torch.zeros(batch_size, self.mem_dim), requires_grad=True)#.cuda()
        self.out = Variable(torch.zeros(batch_size, self.out_dim), requires_grad=True)#.cuda()
        self.rnn_history = []#Variable(torch.zeros())

    def predict(self, input):
        return self.forward(input)

    def graph_compute(self, input, rec_output, memory):

        # Input process
        #block_inp = F.sigmoid(self.w_inp(input) + self.w_rec_inp(rec_output))  # Block Input
        block_inp = torch.sigmoid(self.w_inp(torch.t(input)) + self.w_rec_inp(rec_output))
        inp_gate = torch.sigmoid(self.w_inpgate(torch.t(input)) + self.w_mem_inpgate(memory) + self.w_rec_inpgate(rec_output)) #Input gate

        # Read from memory
        read_gate_out = torch.sigmoid(self.w_readgate(torch.t(input)) + self.w_mem_readgate(memory) + self.w_rec_readgate(rec_output))
        decoded_mem = self.w_decoder(read_gate_out * memory)

        # Compute hidden activation
        hidden_act = decoded_mem + block_inp * inp_gate

        # Update memory
        write_gate_out = torch.sigmoid(self.w_writegate(torch.t(input)) + self.w_mem_writegate(memory) + self.w_rec_writegate(rec_output))  # #Write gate
        encoded_update = torch.tanh(self.w_encoder(hidden_act))
        memory = (1 - write_gate_out) * memory + write_gate_out * encoded_update
        #memory = memory + encoded_update

        return hidden_act, memory

    def forward(self, input):
        # Adaptive components
        self.mem = Variable(torch.zeros(input.shape[1], self.mem_dim), requires_grad=True)#.cuda()
        self.out = Variable(torch.zeros(input.shape[1], self.out_dim), requires_grad=True)#.cuda()

        #print(self.out.shape)
        '''Create history of n time-steps and loop graph_compute n times to generate final output'''
        if not torch.is_tensor(self.rnn_history):
            self.rnn_history = torch.Tensor(np.zeros([input.shape[0], self.hist_steps, input.shape[1]])) #control_inputs, history, batch_size

        # Shift the history and update to latest input
        for i in range(self.hist_steps-1):
            self.rnn_history[:,i,:] = self.rnn_history[:,i+1,:]

        '''Trying to fix batch size change'''
        # the last batch_size can be different
        if input.shape != self.rnn_history[:,-1,:].shape:
            temp = copy.deepcopy(input)
            input = torch.Tensor(np.zeros([self.rnn_history.shape[0], self.rnn_history.shape[2]]))
            input[0:temp.shape[0], 0:temp.shape[1]] = temp

        self.rnn_history[:,-1,:] = input

        #print(self.out.shape)
        # Loop to generate final output
        for i in range(self.hist_steps):
            out, mem = self.graph_compute(self.rnn_history[:,i,:], self.out, self.mem)
            self.out, self.mem = out, mem
            self.out = self.w_hid_out.mm(torch.t(self.out))
            self.out = torch.t(self.out)

        '''Old working code without history'''
        #self.out, self.mem = self.graph_compute(input, self.out, self.mem)
        # Till here, "out" is the hidden_act
        #self.out = self.w_hid_out.mm(torch.t(self.out))
        #self.out = torch.t(self.out)

        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True


#Simulator stuff
def simulator_results(model, filename = 'ColdAir.csv', downsample_rate=25):
    # Import training data and clear away the two top lines
    data = np.loadtxt(filename, delimiter=',', skiprows=2)

    # Splice data (downsample)
    ignore = np.copy(data)
    data = data[0::downsample_rate]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (i != data.shape[0] - 1):
                data[i][j] = ignore[(i * downsample_rate):(i + 1) * downsample_rate,
                             j].sum() / downsample_rate
            else:
                residue = ignore.shape[0] - i * downsample_rate
                data[i][j] = ignore[(i * downsample_rate):i * downsample_rate + residue, j].sum() / residue

    # Normalize between 0-0.99
    normalizer = np.zeros(data.shape[1])
    min = np.zeros(len(data[0]))
    max = np.zeros(len(data[0]))
    for i in range(len(data[0])):
        min[i] = np.amin(data[:, i])
        max[i] = np.amax(data[:, i])
        normalizer[i] = max[i] - min[i] + 0.00001
        data[:, i] = (data[:, i] - min[i]) / normalizer[i]

    print ('TESTING NOW')
    input = np.reshape(data[0], (1, 21))  # First input to the simulatior
    track_target = np.reshape(np.zeros((len(data) - 1) * 19), (19, len(data) - 1))
    track_output = np.reshape(np.zeros((len(data) - 1) * 19), (19, len(data) - 1))

    for example in range(len(data)-1):  # For all training examples
        model_out = model.predict(input)

        # Track index
        for index in range(19):
            track_output[index][example] = model_out[0][index]# * normalizer[index] + min[index]
            track_target[index][example] = data[example+1][index]# * normalizer[index] + min[index]

        # Fill in new input data
        for k in range(len(model_out[0])):
            input[0][k] = model_out[0][k]
        # Fill in two control variables
        input[0][19] = data[example + 1][19]
        input[0][20] = data[example + 1][20]

    for index in range(19):
        plt.plot(track_target[index], 'r--',label='Actual Data: ' + str(index))
        plt.plot(track_output[index], 'b-',label='TF_Simulator: ' + str(index))
        #np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
        #np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
        plt.legend( loc='upper right',prop={'size':6})
        #plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
        #print track_output[index]
        plt.show()


def unsqueeze(array, axis=1):
    if axis == 0: return np.reshape(array, (1, len(array)))
    elif axis == 1: return np.reshape(array, (len(array), 1))


def unpickle(filename):
    with open(filename, 'rb') as f:
        u = cPickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p


def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)
