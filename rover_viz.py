import numpy as np
from core import mod_utils as utils
from envs.env_wrapper import RoverDomainPython
import argparse
import sys
import torch
import torch.nn as nn


RANDOM_BASELINE = False
parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, help='Env to test on?', default='rover_tight')
parser.add_argument('-config', type=str, help='World Setting?', default='6_3')


class ConfigSettings:
	def __init__(self):

		self.env_choice = vars(parser.parse_args())['env']
		config = vars(parser.parse_args())['config']
		self.config = config
		self.reward_scheme = 'mixed'
		#Global subsumes local or vice-versa?
		####################### NIPS EXPERIMENTS SETUP #################
		self.is_lsg = False
		self.is_proxim_rew = True
		self.is_gsl = False
		self.cmd_vel = True

		# ROVER DOMAIN
		if self.env_choice == 'rover_loose' or self.env_choice == 'rover_tight' or self.env_choice == 'rover_trap':  # Rover Domain


			if config == 'two_test':
				# Rover domain
				self.dim_x = self.dim_y = 10
				self.obs_radius = self.dim_x * 10
				self.act_dist = 2
				self.angle_res = 10
				self.num_poi = 2
				self.num_agents = 2
				self.ep_len = 30
				self.poi_rand = 1
				self.coupling = 2
				self.rover_speed = 1
				self.sensor_model = 'closest'

			elif config == 'nav':
				# Rover domain
				self.dim_x = self.dim_y = 30; self.obs_radius = self.dim_x * 10; self.act_dist = 2; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 10
				self.num_agents = 1
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 1


			##########LOOSE##########
			elif config == '3_1':
				# Rover domain
				self.dim_x = self.dim_y = 30; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 3
				self.num_agents = 3
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 1

			##########TIGHT##########
			elif config == '4_2':
				# Rover domain
				self.dim_x = self.dim_y = 20;
				self.obs_radius = self.dim_x * 10;
				self.act_dist = 3;
				self.rover_speed = 1;
				self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 4
				self.num_agents = 4
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 2

			elif config == '6_3':
				# Rover domain
				self.dim_x = self.dim_y = 20; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 4
				self.num_agents = 6
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 3

			elif config == '8_4':
				# Rover domain
				self.dim_x = self.dim_y = 20; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 4
				self.num_agents = 8
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 4

			elif config == '10_5':
				# Rover domain
				self.dim_x = self.dim_y = 20; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 4
				self.num_agents = 10
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 5

			elif config == '12_6':
				# Rover domain
				self.dim_x = self.dim_y = 20; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 4
				self.num_agents = 12
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 6

			elif config == '14_7':
				# Rover domain
				self.dim_x = self.dim_y = 20; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 4
				self.num_agents = 14
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 7

			else:
				sys.exit('Unknown Config')

			#Fix Harvest Period and coupling given some config choices

			if self.env_choice == "rover_trap": self.harvest_period = 3
			else: self.harvest_period = 1

			if self.env_choice == "rover_loose": self.coupling = 1 #Definiton of a Loosely coupled domain

class Parameters:
	def __init__(self):
		self.config = ConfigSettings()
		self.state_dim = int(720 / self.config.angle_res) + 1
		if self.config.cmd_vel: self.state_dim += 2
		self.action_dim = 2

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

args=Parameters()
NUM_EVALS = 1
env = RoverDomainPython(args, NUM_EVALS)


path = 'nets/0_actor_pop20_roll50_envrover_tight_6_3_seed2023-reward'
buffer = torch.load(path)
net = MultiHeadActor(args.state_dim, args.action_dim, 100, args.config.num_agents)
net.load_state_dict(buffer)
net.eval()

joint_state = env.reset()
joint_state = utils.to_tensor(np.array(joint_state))
fitness = [0 for _ in range(NUM_EVALS)]
local_fitness = [0 for _ in range(NUM_EVALS)]

while True: #unless done

	if RANDOM_BASELINE:
		joint_action = [np.random.random((1, args.state_dim))for _ in range(args.config.num_agents)]
	else:
		joint_action = [net.clean_action(joint_state[i, :], head=i).detach().numpy() for i in range(args.config.num_agents)]


	#Bound Action
	joint_action = np.array(joint_action).clip(-1.0, 1.0)
	next_state, reward, done, global_reward = env.step(joint_action)  # Simulate one step in environment

	next_state = utils.to_tensor(np.array(next_state))


	#Grab global reward as fitnesses
	for i, grew in enumerate(global_reward):
		if grew != None:
			fitness[i] = grew
			local_fitness[i] = sum(env.universe[i].cumulative_local)

	joint_state = next_state

	#DONE FLAG IS Received
	if sum(done)==len(done):
		break

best_performant = fitness.index(max(fitness))
#best_performant = local_fitness.index(max(local_fitness))

env.universe[best_performant].render()
env.universe[best_performant].viz()
env.universe[best_performant].viz(save=True, fname='test.png')
print('Global', fitness, fitness[best_performant])
print('Local', local_fitness, local_fitness[best_performant])

#print(fitness)
