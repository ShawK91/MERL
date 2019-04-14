import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch
import torch.nn.functional as F
import torch.nn as nn


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

        # for param in self.parameters():
        #     # torch.nn.init.xavier_normal(param)
        #     # torch.nn.init.orthogonal(param)
        #     # torch.nn.init.sparse(param, sparsity=0.5)
        #     torch.nn.init.kaiming_normal(param)
        self.cuda()

    def reset(self, batch_size):
        # Adaptive components
        self.mem = Variable(torch.zeros(batch_size, self.mem_dim), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(batch_size, self.out_dim), requires_grad=1).cuda()


    def graph_compute(self, input, rec_output, memory):

        # Input process
        block_inp = F.sigmoid(self.w_inp(input) + self.w_rec_inp(rec_output))  # Block Input
        inp_gate = F.sigmoid(self.w_inpgate(input) + self.w_mem_inpgate(memory) + self.w_rec_inpgate(rec_output)) #Input gate

        # Read from memory
        read_gate_out = F.sigmoid(self.w_readgate(input) + self.w_mem_readgate(memory) + self.w_rec_readgate(rec_output))
        decoded_mem = self.w_decoder(read_gate_out * memory)

        # Compute hidden activation
        hidden_act = decoded_mem + block_inp  * inp_gate

        # Update memory
        write_gate_out = F.sigmoid(self.w_writegate(input) + self.w_mem_writegate(memory) + self.w_rec_writegate(rec_output))  # #Write gate
        encoded_update = F.tanh(self.w_encoder(hidden_act))
        memory = (1 - write_gate_out) * memory + write_gate_out * encoded_update
        #memory = memory + encoded_update

        return hidden_act, memory

    def forward(self, input):
        self.out, self.mem = self.graph_compute(input, self.out, self.mem)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super(LSTM, self).__init__()

        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.memory_size = memory_size;
        self.output_size = output_size

        # Input gate
        self.w_inpgate = Parameter(torch.rand(hidden_size, input_size + 1), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand(hidden_size, output_size + 1), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        # Block Input
        self.w_inp = Parameter(torch.rand(hidden_size, input_size + 1), requires_grad=1)
        self.w_rec_inp = Parameter(torch.rand(hidden_size, output_size + 1), requires_grad=1)

        # Read Gate
        self.w_readgate = Parameter(torch.rand(memory_size, input_size + 1), requires_grad=1)
        self.w_rec_readgate = Parameter(torch.rand(memory_size, output_size + 1), requires_grad=1)
        self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Write Gate
        self.w_writegate = Parameter(torch.rand(memory_size, input_size + 1), requires_grad=1)
        self.w_rec_writegate = Parameter(torch.rand(memory_size, output_size + 1), requires_grad=1)
        self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, 1), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, 1), requires_grad=1).cuda()

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

            # Gates to 1
            # self.w_writegate = Parameter(torch.ones(memory_size, input_size), requires_grad=1)
            # self.w_rec_writegate = Parameter(torch.ones(memory_size, output_size), requires_grad=1)
            # self.w_mem_writegate = Parameter(torch.ones(memory_size, memory_size), requires_grad=1)
            # self.w_readgate = Parameter(torch.ones(memory_size, input_size), requires_grad=1)
            # self.w_rec_readgate = Parameter(torch.ones(memory_size, output_size), requires_grad=1)
            # self.w_mem_readgate = Parameter(torch.ones(memory_size, memory_size), requires_grad=1)
            # self.w_inpgate = Parameter(torch.ones(hidden_size, input_size), requires_grad=1)
            # self.w_rec_inpgate = Parameter(torch.ones(hidden_size, output_size), requires_grad=1)
            # self.w_mem_inpgate = Parameter(torch.ones(hidden_size, memory_size), requires_grad=1)

    def prep_bias(self, mat, batch_size):
        return Variable(torch.cat((mat.cpu().data, torch.ones(1, batch_size))).cuda())

    def reset(self, batch_size):
        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, batch_size), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, batch_size), requires_grad=1).cuda()

    def graph_compute(self, input, rec_output, mem, batch_size):
        # Reshape add 1 for bias
        input = self.prep_bias(input, batch_size);
        rec_output = self.prep_bias(rec_output, batch_size)

        # Block Input
        block_inp = F.tanh(self.w_inp.mm(input) + self.w_rec_inp.mm(rec_output))  # + self.w_block_input_bias)

        # Input gate
        inp_gate = F.sigmoid(self.w_inpgate.mm(input) + self.w_mem_inpgate.mm(mem) + self.w_rec_inpgate.mm(
            rec_output))  # + self.w_input_gate_bias)

        # Input out
        inp_out = block_inp * inp_gate

        # Read gate
        read_gate_out = F.sigmoid(
            self.w_readgate.mm(input) + self.w_rec_readgate.mm(rec_output) + self.w_mem_readgate.mm(
                mem))  # + self.w_readgate_bias) * mem

        # Output gate
        out_gate = F.sigmoid(
            self.w_writegate.mm(input) + self.w_mem_writegate.mm(mem) + self.w_rec_writegate.mm(
                rec_output))  # + self.w_writegate_bias)

        # Compute new mem
        mem = inp_out + read_gate_out * mem
        out = out_gate * mem

        return out, mem

    def forward(self, input):
        batch_size = input.data.shape[-1]
        self.out, self.mem = self.graph_compute(input, self.out, self.mem, batch_size)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True
