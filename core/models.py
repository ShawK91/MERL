import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal



LOG_SIG_MAX = 5
LOG_SIG_MIN = -10
epsilon = 1e-6



class MultiHeadActor(nn.Module):
	"""Actor model

		Parameters:
			  args (object): Parameter class
	"""

	def __init__(self, num_inputs, num_actions, hidden_size, num_heads):
		super(MultiHeadActor, self).__init__()

		self.num_heads = num_heads
		self.num_actions = num_actions

		#Trunk
		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)

		#Heads
		self.mean = nn.Linear(hidden_size, num_actions*num_heads)
		self.noise = torch.Tensor(num_actions*num_heads)

		self.apply(weights_init_policy_fn)




	def clean_action(self, state, head=-1):
		"""Method to forward propagate through the actor's graph

			Parameters:
				  input (tensor): states

			Returns:
				  action (tensor): actions


		"""

		x = torch.tanh(self.linear1(state))
		x = torch.tanh(self.linear2(x))
		mean = torch.tanh(self.mean(x))

		if head == -1:
			return mean
		else:
			start = head*self.num_actions
			return mean[:,start:start+self.num_actions]



	def noisy_action(self, state, head=-1):

		x = torch.tanh(self.linear1(state))
		x = torch.tanh(self.linear2(x))
		mean = torch.tanh(self.mean(x))

		action = mean + self.noise.normal_(0., std=0.4)
		if head == -1:
			return action
		else:
			start = head * self.num_actions
			return action[:, start:start + self.num_actions]




	def get_norm_stats(self):
		minimum = min([torch.min(param).item() for param in self.parameters()])
		maximum = max([torch.max(param).item() for param in self.parameters()])
		means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
		mean = sum(means)/len(means)

		return minimum, maximum, mean



class Actor(nn.Module):
	"""Actor model

		Parameters:
			  args (object): Parameter class
	"""

	def __init__(self, num_inputs, num_actions, hidden_size, policy_type):
		super(Actor, self).__init__()

		self.policy_type = policy_type

		if self.policy_type == 'GaussianPolicy':
			self.linear1 = nn.Linear(num_inputs, hidden_size)
			self.linear2 = nn.Linear(hidden_size, hidden_size)

			self.mean_linear = nn.Linear(hidden_size, num_actions)

			self.log_std_linear = nn.Linear(hidden_size, num_actions)

			self.apply(weights_init_policy_fn)

		elif self.policy_type == 'DeterministicPolicy':
			self.linear1 = nn.Linear(num_inputs, hidden_size)
			self.linear2 = nn.Linear(hidden_size, hidden_size)

			self.mean = nn.Linear(hidden_size, num_actions)
			self.noise = torch.Tensor(num_actions)

			self.apply(weights_init_policy_fn)




	def clean_action(self, state, return_only_action=True):
		"""Method to forward propagate through the actor's graph

			Parameters:
				  input (tensor): states

			Returns:
				  action (tensor): actions


		"""
		if self.policy_type == 'GaussianPolicy':
			x = torch.tanh(self.linear1(state))
			x = torch.tanh(self.linear2(x))
			mean = self.mean_linear(x)
			if return_only_action: return torch.tanh(mean)

			log_std = self.log_std_linear(x)
			log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
			return mean, log_std

		elif self.policy_type == 'DeterministicPolicy':
			x = torch.tanh(self.linear1(state))
			x = torch.tanh(self.linear2(x))
			mean = torch.tanh(self.mean(x))
			return mean



	def noisy_action(self, state, return_only_action=True):

		if self.policy_type == 'GaussianPolicy':
			mean, log_std = self.clean_action(state, return_only_action=False)
			std = log_std.exp()
			normal = Normal(mean, std)
			x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
			action = torch.tanh(x_t)

			if return_only_action: return action

			log_prob = normal.log_prob(x_t)
			# Enforcing Action Bound
			log_prob -= torch.log(1 - action.pow(2) + epsilon)
			log_prob = log_prob.sum(-1, keepdim=True)

			#log_prob.clamp(-10, 0)

			return action, log_prob, x_t, mean, log_std

		elif self.policy_type == 'DeterministicPolicy':
			mean = self.clean_action(state)
			action = mean + self.noise.normal_(0., std=0.4)

			if return_only_action: return action
			else: return action, torch.tensor(0.), torch.tensor(0.), mean, torch.tensor(0.)



	def get_norm_stats(self):
		minimum = min([torch.min(param).item() for param in self.parameters()])
		maximum = max([torch.max(param).item() for param in self.parameters()])
		means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
		mean = sum(means)/len(means)

		return minimum, maximum, mean



class ValueNetwork(nn.Module):
	def __init__(self, state_dim, hidden_dim):
		super(ValueNetwork, self).__init__()

		self.linear1 = nn.Linear(state_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, 1)

		self.apply(weights_init_value_fn)

	def forward(self, state):
		x = F.elu(self.linear1(state))
		x = F.elu(self.linear2(x))
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

		self.apply(weights_init_value_fn)

	def forward(self, state, action):
		x1 = torch.cat([state, action], 1)
		x1 = torch.tanh(self.linear1(x1))
		x1 = torch.tanh(self.linear2(x1))
		x1 = self.linear3(x1)

		x2 = torch.cat([state, action], 1)
		x2 = torch.tanh(self.linear4(x2))
		x2 = torch.tanh(self.linear5(x2))
		x2 = self.linear6(x2)

		return x1, x2


class ActualizationNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size):
		super(ActualizationNetwork, self).__init__()

		# Q1 architecture
		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)

		self.apply(weights_init_value_fn)

	def forward(self, state, action):


		x1 = torch.cat([state, action], 1)
		#return self.linear3(self.linear1(x1))
		x1 = F.elu(self.linear1(x1))
		x1 = F.elu(self.linear2(x1))
		x1 = self.linear3(x1)
		return x1




# Initialize Policy weights
def weights_init_policy_fn(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
		torch.nn.init.constant_(m.bias, 0)

# Initialize Value Fn weights
def weights_init_value_fn(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)


class Conv_model(nn.Module):
	def __init__(self, z_dim):
		super(Conv_model, self).__init__()
		self.hw = 5 #Intermediate computation to track the HW dimension

		## Encoder
		self.conv1 = nn.Conv2d(in_channels=18, out_channels=32, kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(32, 48, 3, stride=1)
		self.conv3 = nn.Conv2d(48, 64, 3, stride=1)
		self.fc_encoder = nn.Linear(64 * self.hw * self.hw, z_dim)

		#Policy Net
		self.policy_fc1 = nn.Linear(z_dim, 200)
		self.policy_lnorm1 = torch.LayerNorm(200)
		self.policy_fc2 = nn.Linear(200, 6)

		#Value Net
		self.value_fc1 = nn.Linear(z_dim, 200)
		self.value_lnorm1 = LayerNorm(200)
		self.value_fc2 = nn.Linear(200, 1)


	def encode(self, x):
		h = F.elu(self.conv1(x)); #print(h.shape)
		h = F.elu(self.conv2(h)) ; #print(h.shape)
		h = F.elu(self.conv3(h)); #print(h.shape)
		h = h.view(-1, 64 * self.hw * self.hw); #print(h.shape)
		h = F.elu(self.fc_encoder(h))
		return h


	def policy_value_net(self, z):
		#Compute policy head
		p = F.elu(self.policy_fc1(z))
		p = self.policy_lnorm1(p)
		p = F.softmax(self.policy_fc2(p))

		#Compute value head
		v = F.elu(self.value_fc1(z))
		v = self.value_lnorm1(v)
		v = F.tanh(self.value_fc2(v))
		return p, v


	def forward(self, x):
		z = self.encode(x)
		p, v = self.policy_value_net(z)
		return p, v

	#API FOR MCTS
	def predict(self,x):
		x = torch.Tensor(x).permute([0,3,1,2]).cuda()
		p, v = self.forward(x)
		return utils.to_numpy(p.cpu()), utils.to_numpy(v.cpu())

	def loss_fn(self, p, p_target, v, v_target):
		p_loss = torch.sum(torch.nn.functional.cross_entropy(p, p_target))
		v_loss = torch.sum(torch.nn.functional.mse_loss(v.squeeze(), v_target))
		total_loss = p_loss + v_loss
		return total_loss, p_loss.item(), v_loss.item()


