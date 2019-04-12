from core.agent import Agent, TestAgent
from core.mod_utils import pprint, str2bool
import numpy as np, os, time, torch
from core import mod_utils as utils
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe
import core.mod_utils as mod
import argparse
import random
import threading, sys

parser = argparse.ArgumentParser()
parser.add_argument('-popsize', type=int, help='#Evo Population size', default=0)
parser.add_argument('-rollsize', type=int, help='#Rollout size for agents', default=1)
parser.add_argument('-scheme', type=str, help='Scheme?', default='multipoint')
parser.add_argument('-homogeny', type=str2bool, help='Make the policy homogeneous?', default=True)
parser.add_argument('-alz', type=str2bool, help='Actualize?', default=False)
parser.add_argument('-env', type=str, help='Env to test on?', default='multiwalker')
parser.add_argument('-config', type=str, help='World Setting?', default='2')


parser.add_argument('-evals', type=int, help='#Evals to compute a fitness', default=1)
parser.add_argument('-seed', type=float, help='#Seed', default=2019)
parser.add_argument('-algo', type=str, help='SAC Vs. TD3?', default='TD3')
parser.add_argument('-savetag', help='Saved tag', default='')
parser.add_argument('-gradperstep', type=float, help='gradient steps per frame', default=1.0)
parser.add_argument('-pr', type=float, help='Prioritization?', default=0.0)
parser.add_argument('-use_gpu', type=str2bool, help='USE_GPU?', default=True)


RANDOM_BASELINE = False


class ConfigSettings:
	def __init__(self):

		self.env_choice = vars(parser.parse_args())['env']
		config = vars(parser.parse_args())['config']
		self.config = config

		#ROVER DOMAIN
		if self.env_choice == 'rover_loose' or  self.env_choice == 'rover_right': #Rover Domain
			if config == 'single_test':
				# Rover domain
				self.dim_x = self.dim_y = 10
				self.obs_radius = self.dim_x * 10
				self.act_dist = 2
				self.angle_res = 15
				self.num_poi = 6
				self.num_agents = 1
				self.ep_len = 40
				self.poi_rand = 1
				self.coupling = 1
				self.rover_speed = 1
				self.sensor_model = 'closest'

			elif config == 'mtc_mac':
				# Rover domain
				self.dim_x = self.dim_y = 7
				self.obs_radius = self.dim_x * 10
				self.act_dist = 2
				self.angle_res = 15
				self.num_poi = 2
				self.num_agents = 2
				self.ep_len = 20
				self.poi_rand = 1
				self.coupling = 2
				self.rover_speed = 1
				self.sensor_model = 'closest'

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

			elif config == '15_3':
				# Rover domain
				self.dim_x = self.dim_y = 15
				self.obs_radius = self.dim_x * 10
				self.act_dist = 3
				self.angle_res = 15
				self.num_poi = 9
				self.num_agents = 6
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 3
				self.rover_speed = 1
				self.sensor_model = 'closest'

			elif config == '15_4':
				# Rover domain
				self.dim_x = self.dim_y = 15
				self.obs_radius = self.dim_x * 10;
				self.act_dist = 3
				self.angle_res = 15
				self.num_poi = 9
				self.num_agents = 8
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 4
				self.rover_speed = 1
				self.sensor_model = 'closest'

			elif config == '20_3':
				# Rover domain
				self.dim_x = self.dim_y = 20
				self.obs_radius = self.dim_x * 10;
				self.act_dist = 3
				self.angle_res = 15
				self.num_poi = 9
				self.num_agents = 6
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 3
				self.rover_speed = 1
				self.sensor_model = 'closest'

			elif config == '20_4':
				# Rover domain
				self.dim_x = self.dim_y = 20
				self.obs_radius = self.dim_x * 10;
				self.act_dist = 3
				self.angle_res = 15
				self.num_poi = 9
				self.num_agents = 8
				self.ep_len = 50
				self.poi_rand = 1
				self.coupling = 4
				self.rover_speed = 1
				self.sensor_model = 'closest'

			else:
				sys.exit('Unknown Config')

		#MultiWalker Domain
		elif self.env_choice == 'multiwalker': #MultiWalker Domain
			try: self.num_agents = int(config)
			except:
				sys.exit('Unknown Config Choice for multiwalker env. Choose #walkers')

		else:
			sys.exit('Unknown Environment Choice')


class Parameters:
	def __init__(self):

		# Transitive Algo Params
		self.rollout_size = vars(parser.parse_args())['rollsize']
		self.num_evals = vars(parser.parse_args())['evals']
		self.frames_bound = 100000000
		self.actualize = vars(parser.parse_args())['alz']
		self.priority_rate = vars(parser.parse_args())['pr']
		self.use_gpu = vars(parser.parse_args())['use_gpu']
		self.seed = vars(parser.parse_args())['seed']
		self.is_homogeneous = vars(parser.parse_args())['homogeny']

		# Rover domain
		self.config = ConfigSettings()

		# Fairly Stable Algo params
		self.hidden_size = 200
		self.algo_name = vars(parser.parse_args())['algo']
		self.actor_lr = 1e-4
		self.critic_lr = 1e-4
		self.tau = 1e-3
		self.init_w = True
		self.gradperstep = vars(parser.parse_args())['gradperstep']
		self.gamma = 0.997
		self.batch_size = 512
		self.buffer_size = 1000000 if self.is_homogeneous else 100000
		self.action_loss = False
		self.policy_ups_freq = 2
		self.policy_noise = True
		self.policy_noise_clip = 0.4

		# SAC
		self.alpha = 0.2
		self.target_update_interval = 1

		# NeuroEvolution stuff
		self.popn_size = vars(parser.parse_args())['popsize']
		self.scheme = vars(parser.parse_args())['scheme']  # 'multipoint' vs 'standard'
		self.crossover_prob = 0.1
		self.mutation_prob = 0.9
		self.extinction_prob = 0.005  # Probability of extinction event
		self.extinction_magnitude = 0.5  # Probabilty of extinction for each genome, given an extinction event
		self.weight_clamp = 1000000
		self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform
		self.lineage_depth = 10
		self.ccea_reduction = "leniency"
		self.num_anchors = 5
		self.num_elites = 4
		self.num_blends = int(0.15 * self.popn_size)

		# Dependents
		if self.config.env_choice == 'rover_loose' or  self.config.env_choice == 'rover_right': #Rover Domain
			self.state_dim = int(720 / self.config.angle_res) + 1
			self.action_dim = 2
		elif self.config.env_choice == 'multiwalker': #MultiWalker Domain
			self.state_dim = 32
			self.action_dim = 4
		else:
			sys.exit('Unknown Environment Choice')

		self.num_test = 10
		self.test_gap = 5

		# Save Filenames
		self.savetag = vars(parser.parse_args())['savetag'] + \
		               'pop' + str(self.popn_size) + \
		               '_roll' + str(self.rollout_size) + \
		               '_alz' + str(self.actualize) + \
		               '_env' + str(self.config.env_choice)+'_'+ str(self.config.config)
			# '_pr' + str(self.priority_rate)
		# '_algo' + str(self.algo_name) + \
		# '_evals' + str(self.num_evals) + \
		# '_seed' + str(SEED)

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


class MERL:
	"""Policy Gradient Algorithm main object which carries out off-policy learning using policy gradient
	   Encodes all functionalities for 1. TD3 2. DDPG 3.Trust-region TD3/DDPG 4. Advantage TD3/DDPG

			Parameters:
				args (int): Parameter class with all the parameters

			"""

	def __init__(self, args):
		self.args = args

		######### Initialize the Multiagent Team of agents ########
		if self.args.is_homogeneous:
			self.agents = [Agent(self.args, id)]
		else:
			self.agents = [Agent(self.args, id) for id in range(self.args.config.num_agents)]
		self.test_agent = TestAgent(self.args, 991)

		###### Buffer and Model Bucket as references to the corresponding agent's attributes ####
		self.buffer_bucket = [ag.buffer.tuples for ag in self.agents]
		self.popn_bucket = [ag.popn for ag in self.agents]
		self.rollout_bucket = [ag.rollout_actor for ag in self.agents]
		self.test_bucket = self.test_agent.rollout_actor

		######### EVOLUTIONARY WORKERS ############
		if self.args.popn_size > 0:
			self.evo_task_pipes = [Pipe() for _ in range(args.popn_size * args.num_evals)]
			self.evo_result_pipes = [Pipe() for _ in range(args.popn_size * args.num_evals)]
			self.evo_workers = [Process(target=rollout_worker, args=(
			self.args, i, 'evo', self.evo_task_pipes[i][1], self.evo_result_pipes[i][0],
			self.buffer_bucket, self.popn_bucket, True, RANDOM_BASELINE)) for i in
			                    range(args.popn_size * args.num_evals)]
			for worker in self.evo_workers: worker.start()

		######### POLICY GRADIENT WORKERS ############
		if self.args.rollout_size > 0:
			self.pg_task_pipes = Pipe()
			self.pg_result_pipes = Pipe()
			self.pg_workers = [
				Process(target=rollout_worker, args=(self.args, 0, 'pg', self.pg_task_pipes[1], self.pg_result_pipes[0],
				                                     self.buffer_bucket, self.rollout_bucket,
				                                     self.args.rollout_size > 0, RANDOM_BASELINE))]
			for worker in self.pg_workers: worker.start()

		######### TEST WORKERS ############
		self.test_task_pipes = Pipe()
		self.test_result_pipes = Pipe()
		self.test_workers = [Process(target=rollout_worker,
		                             args=(self.args, 0, 'test', self.test_task_pipes[1], self.test_result_pipes[0],
		                                   None, self.test_bucket, False, RANDOM_BASELINE))]
		for worker in self.test_workers: worker.start()

		#### STATS AND TRACKING WHICH ROLLOUT IS DONE ######
		self.best_score = -999;
		self.total_frames = 0;
		self.gen_frames = 0;
		self.test_trace = []

	def make_teams(self, num_agents, popn_size, num_evals):

		temp_inds = []
		for _ in range(num_evals): temp_inds += list(range(popn_size))

		all_inds = [temp_inds[:] for _ in range(num_agents)]
		# for _ in range(num_evals):
		# 	for ag in range(num_agents):
		# 		all_inds[ag] += list(range(popn_size))

		for entry in all_inds: random.shuffle(entry)

		teams = [[entry[i] for entry in all_inds] for i in range(popn_size * num_evals)]

		return teams

	def train(self, gen, test_tracker):
		"""Main training loop to do rollouts and run policy gradients

			Parameters:
				gen (int): Current epoch of training

			Returns:
				None
		"""

		# Test Rollout
		if gen % self.args.test_gap == 0:
			self.test_agent.make_champ_team(self.agents)  # Sync the champ policies into the TestAgent
			self.test_task_pipes[0].send("START")

		# Figure out teams for Coevolution
		if self.args.is_homogeneous:
			teams = [[i] for i in list(range(
				args.popn_size))]  # Homogeneous case is just the popn as a list of lists to maintain compatibility
		else:
			teams = self.make_teams(args.config.num_agents, args.popn_size, args.num_evals)  # Heterogeneous Case

		########## START EVO ROLLOUT ##########
		if self.args.popn_size > 0:
			for pipe, team in zip(self.evo_task_pipes, teams):
				pipe[0].send(team)

		########## START POLICY GRADIENT ROLLOUT ##########
		if self.args.rollout_size > 0 and not RANDOM_BASELINE:
			# Synch pg_actors to its corresponding rollout_bucket
			for agent in self.agents: agent.update_rollout_actor()

			# Start rollouts using the rollout actors
			self.pg_task_pipes[0].send('START')  # Index 0 for the Rollout bucket

			############ POLICY GRADIENT UPDATES #########
			# Spin up threads for each agent
			threads = [threading.Thread(target=agent.update_parameters, args=()) for agent in self.agents]

			# Start threads
			for thread in threads: thread.start()

			# Join threads
			for thread in threads: thread.join()

		all_fits = []
		####### JOIN EVO ROLLOUTS ########
		if self.args.popn_size > 0:
			for pipe in self.evo_result_pipes:
				entry = pipe[1].recv()
				team = entry[0];
				fitness = entry[1][0];
				frames = entry[2]

				for agent_id, popn_id in enumerate(team): self.agents[agent_id].fitnesses[popn_id].append(
					utils.list_mean(fitness))  ##Assign
				all_fits.append(utils.list_mean(fitness))
				self.total_frames += frames

		####### JOIN PG ROLLOUTS ########
		pg_fits = []
		if self.args.rollout_size > 0 and not RANDOM_BASELINE:
			entry = self.pg_result_pipes[1].recv()
			pg_fits = entry[1][0]
			self.total_frames += entry[2]

		####### JOIN TEST ROLLOUTS ########
		test_fits = []
		if gen % self.args.test_gap == 0:
			entry = self.test_result_pipes[1].recv()
			test_fits = entry[1][0]
			test_tracker.update([mod.list_mean(test_fits)], self.total_frames)
			self.test_trace.append(mod.list_mean(test_fits))

		# Evolution Step
		for agent in self.agents:
			agent.evolve()

		# #Save models periodically
		# if gen % 20 == 0:
		#     for rover_id in range(self.args.num_rover):
		#         torch.save(self.agents[rover_id].critic.state_dict(), self.args.model_save + self.args.critic_fname + '_'+ str(rover_id))
		#         torch.save(self.agents[rover_id].actor.state_dict(), self.args.model_save + self.args.actor_fname + '_'+ str(rover_id))
		#     print("Models Saved")

		return all_fits, pg_fits, test_fits


if __name__ == "__main__":
	args = Parameters()  # Create the Parameters class
	test_tracker = utils.Tracker(args.metric_save, [args.log_fname], '.csv')  # Initiate tracker
	torch.manual_seed(args.seed);
	np.random.seed(args.seed);
	random.seed(args.seed)  # Seeds

	# INITIALIZE THE MAIN AGENT CLASS
	ai = MERL(args)
	print('Running ', args.config.env_choice, 'with config ', args.config.config, ' State_dim:', args.state_dim, 'Action_dim', args.action_dim)
	time_start = time.time()

	###### TRAINING LOOP ########
	for gen in range(1, args.frames_bound):  # RUN VIRTUALLY FOREVER

		# ONE EPOCH OF TRAINING
		popn_fits, pg_fits, test_fits = ai.train(gen, test_tracker)

		# PRINT PROGRESS
		print('Ep:/Frames', gen, '/', ai.total_frames, 'Popn stat:', mod.list_stat(popn_fits), 'PG_stat:',
		      mod.list_stat(pg_fits),
		      'Test_trace:', [pprint(i) for i in ai.test_trace[-5:]], 'FPS:',
		      pprint(ai.total_frames / (time.time() - time_start)), 'Evo', args.scheme, 'Homogeny:', args.is_homogeneous
		      )

		if gen % 5 == 0:
			print()
			print('Test_stat:', mod.list_stat(test_fits), 'SAVETAG:  ', args.savetag)
			print('Weight Stats: min/max/average', pprint(ai.test_bucket[0].get_norm_stats()))
			print('Buffer Lens:', [ag.buffer.__len__() for ag in ai.agents])
			print()

		if gen % 10 == 0 and args.rollout_size > 0:
			print()
			print('Q', pprint(ai.agents[0].algo.q))
			print('Q_loss', pprint(ai.agents[0].algo.q_loss))
			print('Policy', pprint(ai.agents[0].algo.policy_loss))
			if args.algo_name == 'TD3':
				print('Alz_Score', pprint(ai.agents[0].algo.alz_score))
				print('Alz_policy', pprint(ai.agents[0].algo.alz_policy))

			if args.algo_name == 'SAC':
				print('Val', pprint(ai.agents[0].algo.val))
				print('Val_loss', pprint(ai.agents[0].algo.value_loss))
				print('Mean_loss', pprint(ai.agents[0].algo.mean_loss))
				print('Std_loss', pprint(ai.agents[0].algo.std_loss))

			# Buffer Stats
			print('R_mean:', [agent.buffer.rstats['mean'] for agent in ai.agents])
			print('G_mean:', [agent.buffer.gstats['mean'] for agent in ai.agents])

			print('########################################################################')
