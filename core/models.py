import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    """Actor model

        Parameters:
              args (object): Parameter class
    """

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        #self.wwid = torch.Tensor([wwid])
        l1 = 400; l2 = 300

        # Construct Hidden Layer 1
        self.f1 = nn.Linear(state_dim, l1)
        self.ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.f2 = nn.Linear(l1, l2)
        self.ln2 = nn.LayerNorm(l2)

        #Out
        self.w_out = nn.Linear(l2, action_dim)

    def forward(self, input):
        """Method to forward propagate through the actor's graph

            Parameters:
                  input (tensor): states

            Returns:
                  action (tensor): actions


        """
        #Hidden Layer 1
        out = F.elu(self.f1(input))
        out = self.ln1(out)

        #Hidden Layer 2
        out = F.elu(self.f2(out))
        out = self.ln2(out)

        #Out
        return torch.tanh(self.w_out(out))


class Critic(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        l1 = 400; l2 = 300

        ######################## Q1 Head ##################
        # Construct Hidden Layer 1 with state
        self.q1f1 = nn.Linear(state_dim + action_dim, l1)
        self.q1ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.q1f2 = nn.Linear(l1, l2)
        self.q1ln2 = nn.LayerNorm(l2)

        #Out
        self.q1out = nn.Linear(l2, 1)


        ######################## Q2 Head ##################
        # Construct Hidden Layer 1 with state
        self.q2f1 = nn.Linear(state_dim + action_dim, l1)
        self.q2ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.q2f2 = nn.Linear(l1, l2)
        self.q2ln2 = nn.LayerNorm(l2)

        #Out
        self.q2out = nn.Linear(l2, 1)

        ######################## Value Head ##################
        # Construct Hidden Layer 1 with
        self.vf1 = nn.Linear(state_dim, l1)
        self.vln1 = nn.LayerNorm(l1)

        # Hidden Layer 2
        self.vf2 = nn.Linear(l1, l2)
        self.vln2 = nn.LayerNorm(l2)

        # Out
        self.vout = nn.Linear(l2, 1)





    def forward(self, obs, action):
        """Method to forward propagate through the critic's graph

             Parameters:
                   input (tensor): states
                   input (tensor): actions

             Returns:
                   Q1 (tensor): Qval 1
                   Q2 (tensor): Qval 2
                   V (tensor): Value



         """

        #Concatenate observation+action as critic state
        state = torch.cat([obs, action], 1)

        ###### Q1 HEAD ####
        q1 = F.elu(self.q1f1(state))
        q1 = self.q1ln1(q1)
        q1 = F.elu(self.q1f2(q1))
        q1 = self.q1ln2(q1)
        q1 = self.q1out(q1)

        ###### Q2 HEAD ####
        q2 = F.elu(self.q2f1(state))
        q2 = self.q2ln1(q2)
        q2 = F.elu(self.q2f2(q2))
        q2 = self.q2ln2(q2)
        q2 = self.q2out(q2)

        ###### Value HEAD ####
        v = F.elu(self.vf1(obs))
        v = self.vln1(v)
        v = F.elu(self.vf2(v))
        v = self.vln2(v)
        v = self.vout(v)


        return q1, q2, v


################### SAC Parts #############
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, 1)

        self.apply(weights_init)

    def forward(self, state, action):
        x1 = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = torch.cat([state, action], 1)
        x2 = F.relu(self.linear4(x2))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, wwid):
        super(GaussianPolicy, self).__init__()

        self.wwid = torch.Tensor([wwid])

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)

        self.apply(weights_init)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def evaluate(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, x_t, mean, log_std

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = F.tanh(self.mean(x))
        return mean


    def evaluate(self, state):
        mean = self.forward(state)
        action = mean + self.noise.normal_(0., std=0.015)
