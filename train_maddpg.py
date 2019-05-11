from envs.env_wrapper import RoverDomainPython, MotivateDomain, MultiWalker
from maddpg.MADDPG import MADDPG
import numpy as np, sys, os
import torch as th, argparse
#import visdom
import core.mod_utils as utils
from core import mod_utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, help='Env to test on?', default='rover_tight')
parser.add_argument('-config', type=str, help='World Setting?', default='nav')
parser.add_argument('-frames', type=float, help='Frames in millions?', default=50)
parser.add_argument('-seed', type=int, help='#Seed', default=2019)
parser.add_argument('-savetag', help='Saved tag', default='')


class ConfigSettings:
	def __init__(self):

		self.env_choice = vars(parser.parse_args())['env']
		config = vars(parser.parse_args())['config']
		self.config = config
		self.cmd_vel = 1

		#Global subsumes local or vice-versa?
		self.is_gsl = False
		self.is_lsg = False

		# ROVER DOMAIN
		if self.env_choice == 'rover_loose' or self.env_choice == 'rover_tight' or self.env_choice == 'rover_trap':  # Rover Domain

			if config == 'single_test':
				# Rover domain
				self.dim_x = self.dim_y = 8
				self.obs_radius = self.dim_x * 8
				self.act_dist = 2
				self.angle_res = 5
				self.num_poi = 2
				self.num_agents = 1
				self.ep_len = 20
				self.poi_rand = 0
				self.coupling = 1
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

			elif config == 'two_test':
				# Rover domain
				self.dim_x = self.dim_y = 10
				self.obs_radius = self.dim_x * 10
				self.act_dist = 2
				self.angle_res = 15
				self.num_poi = 6
				self.num_agents = 2
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 2
				self.rover_speed = 1
				self.sensor_model = 'closest'

			elif config == '3_1':
				# Rover domain
				self.dim_x = self.dim_y = 30; self.obs_radius = self.dim_x * 10; self.act_dist = 3; self.rover_speed = 1; self.sensor_model = 'closest'
				self.angle_res = 10
				self.num_poi = 3
				self.num_agents = 3
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 1


			else:
				sys.exit('Unknown Config')

			#Fix Harvest Period and coupling given some config choices

			if self.env_choice == "rover_trap": self.harvest_period = 3
			else: self.harvest_period = 1

			if self.env_choice == "rover_loose": self.coupling = 1 #Definiton of a Loosely coupled domain




		elif self.env_choice == 'motivate':  # Rover Domain
			# Motivate domain
			self.dim_x = self.dim_y = 20
			self.obs_radius = self.dim_x * 10
			self.act_dist = 1.5
			self.angle_res = 10
			self.num_poi = 2
			self.num_agents = 2
			self.ep_len = 20
			self.poi_rand = 0
			self.coupling = 1
			self.rover_speed = 1
			self.sensor_model = 'closest'


		# MultiWalker Domain
		elif self.env_choice == 'multiwalker':  # MultiWalker Domain
			try:
				self.num_agents = int(config)
			except:
				sys.exit('Unknown Config Choice for multiwalker env. Choose #walkers')

		# Cassie Domain
		elif self.env_choice == 'cassie':  # Cassie Domain
			self.num_agents = 1

		# Hyper Domain
		elif self.env_choice == 'hyper':  # Hyper Domain
			self.num_agents = 1
			self.target_sensor = 11
			self.run_time = 300
			self.sensor_noise = 0.1
			self.reconf_shape = 2
			self.num_profiles = 3 #only applicable for some reconf_shapes


		else:
			sys.exit('Unknown Environment Choice')

class Parameters:
	def __init__(self):

		# Transitive Algo Params
		self.frames_bound = int(vars(parser.parse_args())['frames'] * 1000000)
		self.seed = vars(parser.parse_args())['seed']

		# Env domain
		self.config = ConfigSettings()


		# Dependents
		if self.config.env_choice == 'rover_loose' or self.config.env_choice == 'rover_tight' or self.config.env_choice == 'rover_trap':  # Rover Domain
			self.state_dim = int(720 / self.config.angle_res) + 3
			self.action_dim = 2
		elif self.config.env_choice == 'motivate':  # MultiWalker Domain
			self.state_dim = int(720 / self.config.angle_res) + 3
			self.action_dim = 1
		elif self.config.env_choice == 'multiwalker':  # MultiWalker Domain
			self.state_dim = 33
			self.action_dim = 4
		elif self.config.env_choice == 'cassie':  # Cassie Domain
			self.state_dim = 80
			self.action_dim = 10
		elif self.config.env_choice == 'hyper':  # Cassie Domain
			self.state_dim = 20
			self.action_dim = 2
		else:
			sys.exit('Unknown Environment Choice')


		self.num_test = 10
		self.test_gap = 5

		# Save Filenames
		self.savetag = vars(parser.parse_args())['savetag'] + \
					   '_env' + str(self.config.env_choice) + '_' + str(self.config.config) + \
					   '_seed' + str(self.seed) + \
					   '_maddpg'


		self.save_foldername = 'R_MERL/'
		if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
		self.metric_save = self.save_foldername + 'metrics/'
		self.model_save = self.save_foldername + 'models/'
		self.aux_save = self.save_foldername + 'auxiliary/'
		if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
		if not os.path.exists(self.metric_save): os.makedirs(self.metric_save)
		if not os.path.exists(self.model_save): os.makedirs(self.model_save)
		if not os.path.exists(self.aux_save): os.makedirs(self.aux_save)

		self.critic_fname = 'critic_' + self.savetag
		self.actor_fname = 'actor_' + self.savetag
		self.log_fname = 'reward_' + self.savetag
		self.best_fname = 'best_' + self.savetag

def test_policy(world, maddpg):
	global_reward = 0.0; local_reward = 0.0
	for eval in range(10):
		obs = world.reset()
		obs = utils.to_tensor(obs).cuda()
		total_reward = 0.0
		while True:
			action = maddpg.select_action(obs, noise=False).data.cpu()
			obs_, reward, done, grew = world.step(np.reshape(action.numpy(), (args.config.num_agents, 1, args.action_dim)))
			obs_ = th.from_numpy(obs_).float()
			next_obs = obs_.cuda()

			total_reward += reward.sum()
			obs = next_obs
			local_reward += reward.sum()/world.args.config.num_agents

			if sum(done)==len(done):
				global_reward += sum(grew)
				break
	global_reward /= 10.0
	local_reward/=10.0

	return global_reward, local_reward



args = Parameters()
test_tracker = utils.Tracker(args.metric_save, [args.log_fname], '.csv')
test_local_tracker = utils.Tracker(args.metric_save, [args.log_fname+'_local'], '.csv')

# do notrender the scene
world = RoverDomainPython(args, 1)
#vis = visdom.Visdom(port=5274)

np.random.seed(args.seed)
th.manual_seed(args.seed)
#world.seed(1234)
n_agents = args.config.num_agents
n_states = args.state_dim
n_actions = args.action_dim
capacity = 1000000
batch_size = 512

episodes_before_train = 50

win = None
param = None

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
				episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
frames = 0
for i_episode in range(1, 10000000000):
	obs = world.reset()
	obs = utils.to_tensor(obs).cuda()
	total_reward = 0.0
	#rr = np.zeros((n_agents,))
	while True:
		# render every 100 episodes to speed up training
		# if i_episode % 100 == 0 and e_render:
		# 	world.render()
		#obs = obs.type(FloatTensor)
		frames += 1
		action = maddpg.select_action(obs).data.cpu()
		obs_, reward, done, global_reward = world.step(np.reshape(action.numpy(), (args.config.num_agents, 1, args.action_dim)))


		reward = th.FloatTensor(reward).type(FloatTensor)
		#obs_ = np.stack(obs_)
		obs_ = th.from_numpy(obs_).float()

		if sum(done)==len(done): next_obs = None
		else: next_obs = obs_.cuda()

		total_reward += reward.sum()
		#rr += reward.cpu().numpy()

		maddpg.memory.push(obs.data.squeeze(1), action, None if sum(done)==len(done) else next_obs.squeeze(1), reward)
		obs = next_obs

		c_loss, a_loss = maddpg.update_policy()
		if sum(done)==len(done):
			break

	maddpg.episode_done += 1
	#print('Episode: %d, Frames: %d, reward = %f' % (i_episode, frames, total_reward), global_reward)

	if i_episode % 10 == 0:
		test_global, test_local = test_policy(world, maddpg)
		test_tracker.update([test_global], frames)
		test_local_tracker.update([test_local], frames)
		print('Episode: %d, Frames: %d, reward = %f' % (i_episode, frames, test_local), test_global)
	if frames > args.frames_bound: break



	# if maddpg.episode_done == maddpg.episodes_before_train:
	# 	print('training now begins...')
	# 	print('MADDPG on WaterWorld\n' +
	# 		  'scale_reward=%f\n' % scale_reward +
	# 		  'agent=%d' % n_agents +
	# 		  ', coop=%d' % n_coop +
	# 		  ' \nlr=0.001, 0.0001, sensor_range=0.3\n' +
	# 		  'food=%f, poison=%f, encounter=%f' % (
	# 			  food_reward,
	# 			  poison_reward,
	# 			  encounter_reward))

	# if win is None:
	#     win = vis.line(X=np.arange(i_episode, i_episode+1),
	#                    Y=np.array([
	#                        np.append(total_reward, rr)]),    if param is None:
	#     param = vis.line(X=np.arange(i_episode, i_episode+1),
	#                      Y=np.array([maddpg.var[0]]),
	#                      opts=dict(
	#                          ylabel='Var',
	#                          xlabel='Episode',
	#                          title='MADDPG on WaterWorld: Exploration',
	#                          legend=['Variance']))
	# else:
	# 	pass
		# vis.line(X=np.array([i_episode]),
		#          Y=np.array([maddpg.var[0]]),
		#          win=param,
		#          update='append')
	#                    opts=dict(
	#                        ylabel='Reward',
	#                        xlabel='Episode',
	#                        title='MADDPG on WaterWorld_mod\n' +
	#                        'agent=%d' % n_agents +
	#                        ', coop=%d' % n_coop +
	#                        ', sensor_range=0.2\n' +
	#                        'food=%f, poison=%f, encounter=%f' % (
	#                            food_reward,
	#                            poison_reward,
	#                            encounter_reward),
	#                        legend=['Total'] +
	#                        ['Agent-%d' % i for i in range(n_agents)]))
	# else:
	#     vis.line(X=np.array(
	#         [np.array(i_episode).repeat(n_agents+1)]),
	#              Y=np.array([np.append(total_reward,
	#                                    rr)]),
	#              win=win,
	#              update='append')
	# if param is None:
	#     param = vis.line(X=np.arange(i_episode, i_episode+1),
	#                      Y=np.array([maddpg.var[0]]),
	#                      opts=dict(
	#                          ylabel='Var',
	#                          xlabel='Episode',
	#                          title='MADDPG on WaterWorld: Exploration',
	#                          legend=['Variance']))
	# else:
	#     vis.line(X=np.array([i_episode]),
	#              Y=np.array([maddpg.var[0]]),
	#              win=param,
	#              update='append')

world.close()
