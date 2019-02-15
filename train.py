from core.agent import Agent, TestAgent
from core.mod_utils import list_mean, pprint, str2bool
import numpy as np, os, time, random, torch
from core import mod_utils as utils
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
import core.mod_utils as mod
import argparse
import random
import threading

DEBUG = True
RANDOM_BASELINE = False



#ARGPARSE
if DEBUG:
	parser = argparse.ArgumentParser()
	parser.add_argument('-popsize', type=int,  help='#Evo Population size',  default=5)
	parser.add_argument('-rollsize', type=int,  help='#Rollout size for agents',  default=5)
	parser.add_argument('-pg', type=str2bool,  help='#Use PG?',  default=1)
	parser.add_argument('-evals', type=int,  help='#Evals to compute a fitness',  default=5)

	parser.add_argument('-seed', type=float,  help='#Seed',  default=2019)
	parser.add_argument('-dim', type=int,  help='World dimension',  default=7)
	parser.add_argument('-agents', type=int,  help='#agents',  default=2)
	parser.add_argument('-pois', type=int,  help='#POIs',  default=3)
	parser.add_argument('-coupling', type=int,  help='Coupling',  default=1)
	parser.add_argument('-eplen', type=int,  help='eplen',  default=50)
	parser.add_argument('-angle_res', type=int,  help='angle resolution',  default=15)
	parser.add_argument('-randpoi', type=str2bool,  help='#Ranodmize POI initialization?',  default=1)
	parser.add_argument('-sensor_model', type=str,  help='Sensor model: closest vs density?',  default='closest')
	parser.add_argument('-savetag', help='Saved tag',  default='')
	parser.add_argument('-algo', type=str,  help='SAC Vs. TD3?',  default='SAC')

else:
	parser = argparse.ArgumentParser()
	parser.add_argument('-popsize', type=int,  help='#Evo Population size',  default=10)
	parser.add_argument('-rollsize', type=int,  help='#Rollout size for agents',  default=10)
	parser.add_argument('-pg', type=str2bool,  help='#Use PG?',  default=1)
	parser.add_argument('-evals', type=int,  help='#Evals to compute a fitness',  default=5)

	parser.add_argument('-seed', type=float,  help='#Seed',  default=2019)
	parser.add_argument('-dim', type=int,  help='World dimension',  default=15)
	parser.add_argument('-agents', type=int,  help='#agents',  default=1)
	parser.add_argument('-pois', type=int,  help='#POIs',  default=4)
	parser.add_argument('-coupling', type=int,  help='Coupling',  default=1)
	parser.add_argument('-eplen', type=int,  help='eplen',  default=30)
	parser.add_argument('-angle_res', type=int,  help='angle resolution',  default=30)
	parser.add_argument('-randpoi', type=str2bool,  help='#Ranodmize POI initialization?',  default=1)
	parser.add_argument('-sensor_model', type=str,  help='Sensor model: closest vs density?',  default='closest')
	parser.add_argument('-algo', type=str,  help='SAC Vs. TD3?',  default='SAC')
	parser.add_argument('-savetag', help='Saved tag',  default='')


SEED = vars(parser.parse_args())['seed']
USE_PG = vars(parser.parse_args())['pg']
CUDA = True
TEST_GAP = 5



class WorldSettings:
	def __init__(self, roverdomainid):

		if roverdomainid == 1:
			# Rover domain
			self.dim_x = self.dim_y = 10
			self.obs_radius = self.dim_x * 10;
			self.act_dist = 2;
			self.angle_res = 20
			self.num_poi = 3
			self.num_agents = 1
			self.ep_len = 40
			self.poi_rand = 1
			self.coupling = 1
			self.rover_speed = 1
			self.sensor_model = 'closest'



class Parameters:
	def __init__(self):

		#Meta
		self.rollout_size = vars(parser.parse_args())['rollsize']
		self.popn_size = vars(parser.parse_args())['popsize']
		self.num_evals = vars(parser.parse_args())['evals']
		self.frames_bound = 100000000


		#Rover domain
		self.dim_x = self.dim_y = vars(parser.parse_args())['dim']; self.obs_radius = self.dim_x * 2; self.act_dist = 2; self.angle_res = vars(parser.parse_args())['angle_res']
		self.num_poi = vars(parser.parse_args())['pois']; self.num_agents = vars(parser.parse_args())['agents']; self.ep_len = vars(parser.parse_args())['eplen']
		self.poi_rand = vars(parser.parse_args())['randpoi']; self.coupling = vars(parser.parse_args())['coupling']; self.rover_speed = 1
		self.sensor_model = vars(parser.parse_args())['sensor_model']  #Closest VS Density


		#TD3 params
		self.hidden_size = 100
		self.algo_name = vars(parser.parse_args())['algo']
		self.actor_lr = 1e-3
		self.critic_lr = 1e-3
		self.tau = 5e-3
		self.init_w = True
		self.gradperstep = 1.0
		self.gamma = 0.997
		self.batch_size = 128
		self.buffer_size = 100000
		self.updates_per_step = 1
		self.action_loss = False
		self.policy_ups_freq = 2
		self.policy_noise = True
		self.policy_noise_clip = 0.4

		# NeuroEvolution stuff
		self.elite_fraction = 0.2
		self.crossover_prob = 0.15
		self.mutation_prob = 0.90
		self.extinction_prob = 0.005  # Probability of extinction event
		self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
		self.weight_magnitude_limit = 10000000
		self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform

		#Dependents
		self.state_dim = int(720 / self.angle_res)
		self.action_dim = 2
		self.num_test = 10

		#Save Filenames
		self.savetag = vars(parser.parse_args())['savetag'] + \
				   '_pop' + str(self.popn_size) + \
				   '_roll' + str(self.rollout_size) + \
				   '_evals' + str(self.num_evals) + \
					'_algo' + str(self.algo_name) + \
					'_poi_rand' + str(self.poi_rand) + \
					'_dim' + str(self.dim_x) + \
				   '_angle' + str(self.angle_res) + \
				   '_couple' + str(self.coupling) + \
				   '_eplen' + str(self.ep_len) + \
				   '#pois' + str(self.num_poi) + \
				   '_#agents' + str(self.num_agents) + \
				   '_sensor' + str(self.sensor_model) + \
				   '_use_pg' + str(USE_PG) + \
				   '_seed' + str(SEED)



		self.save_foldername = 'R_MERL/'
		if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
		self.metric_save = self.save_foldername + 'metrics/'
		self.model_save = self.save_foldername + 'models/'
		self.aux_save = self.save_foldername + 'auxiliary/'
		if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
		if not os.path.exists(self.metric_save): os.makedirs(self.metric_save)
		if not os.path.exists(self.model_save): os.makedirs(self.model_save)
		if not os.path.exists(self.aux_save): os.makedirs(self.aux_save)



		self.critic_fname = 'critic_' +self.savetag
		self.actor_fname = 'actor_'+ self.savetag
		self.log_fname = 'reward_'+  self.savetag
		self.best_fname = 'best_'+ self.savetag

		#Unit tests (Simply changes the rover/poi init locations)
		self.unit_test = 0 #0: None
						   #1: Single Agent
						   #2: Multiagent 2-coupled



class MERL:
	"""Policy Gradient Algorithm main object which carries out off-policy learning using policy gradient
	   Encodes all functionalities for 1. TD3 2. DDPG 3.Trust-region TD3/DDPG 4. Advantage TD3/DDPG

			Parameters:
				args (int): Parameter class with all the parameters

			"""

	def __init__(self, args):
		self.args = args


		######### Initialize the Multiagent Team of agents ########
		self.agents = [Agent(self.args, id) for id in range(self.args.num_agents)]
		self.test_agent = TestAgent(self.args, 991)


		###### Buffer and Model Bucket as references to the corresponding agent's attributes ####
		self.buffer_bucket = [ag.buffer.tuples for ag in self.agents]
		self.popn_bucket = [ag.popn for ag in self.agents]
		self.rollout_bucket = [ag.rollout_actor for ag in self.agents]
		self.test_bucket = self.test_agent.rollout_actor


		######### EVOLUTIONARY WORKERS ############
		self.evo_task_pipes = [Pipe() for _ in range(args.popn_size*args.num_evals)]
		self.evo_result_pipes = [Pipe() for _ in range(args.popn_size*args.num_evals)]
		self.evo_workers = [Process(target=rollout_worker, args=(self.args, i, 'evo', self.evo_task_pipes[i][1], self.evo_result_pipes[i][0],
																   self.buffer_bucket, self.popn_bucket, USE_PG, RANDOM_BASELINE)) for i in range(args.popn_size*args.num_evals)]
		for worker in self.evo_workers: worker.start()


		######### POLICY GRADIENT WORKERS ############
		self.pg_task_pipes = Pipe()
		self.pg_result_pipes = Pipe()
		self.pg_workers = [Process(target=rollout_worker, args=(self.args, 0, 'pg', self.pg_task_pipes[1], self.pg_result_pipes[0],
																   self.buffer_bucket, self.rollout_bucket, USE_PG, RANDOM_BASELINE))]
		for worker in self.pg_workers: worker.start()

		######### TEST WORKERS ############
		self.test_task_pipes = Pipe()
		self.test_result_pipes = Pipe()
		self.test_workers = [Process(target=rollout_worker, args=(self.args, 0, 'test', self.test_task_pipes[1], self.test_result_pipes[0],
																   None, self.test_bucket, False, RANDOM_BASELINE))]
		for worker in self.test_workers: worker.start()


		#### STATS AND TRACKING WHICH ROLLOUT IS DONE ######
		self.best_score = -999; self.total_frames = 0; self.gen_frames = 0


	def make_teams(self, num_agents, popn_size, num_evals):

		temp_inds = []
		for _ in range(num_evals): temp_inds += list(range(popn_size))


		all_inds = [temp_inds[:] for _ in range(num_agents)]
		# for _ in range(num_evals):
		# 	for ag in range(num_agents):
		# 		all_inds[ag] += list(range(popn_size))

		for entry in all_inds: random.shuffle(entry)

		teams = [[entry[i] for entry in all_inds] for i in range(popn_size*num_evals)]

		return teams


	def train(self, gen, test_tracker):
		"""Main training loop to do rollouts and run policy gradients

			Parameters:
				gen (int): Current epoch of training

			Returns:
				None
		"""


		#Test Rollout
		if gen % TEST_GAP == 0:
			self.test_agent.make_champ_team(self.agents) #Sync the champ policies into the TestAgent
			self.test_task_pipes[0].send("START")


		#Figure out teams for Coevolution
		teams = self.make_teams(args.num_agents, args.popn_size, args.num_evals)

		########## START EVO ROLLOUT ##########
		for pipe, team in zip(self.evo_task_pipes, teams):
			pipe[0].send(team)


		########## START POLICY GRADIENT ROLLOUT ##########
		if USE_PG and not RANDOM_BASELINE:
			#Synch pg_actors to its corresponding rollout_bucket
			for agent in self.agents: agent.update_rollout_actor()

			#Start rollouts using the rollout actors
			self.pg_task_pipes[0].send('START') #Index 0 for the Rollout bucket


			############ POLICY GRADIENT UPDATES #########
			# Spin up threads for each agent
			threads = [threading.Thread(target=agent.update_parameters, args=()) for agent in self.agents]

			# Start threads
			for thread in threads: thread.start()

			# Join threads
			for thread in threads: thread.join()


		all_fits = []
		####### JOIN EVO ROLLOUTS ########
		for pipe in self.evo_result_pipes:
			entry = pipe[1].recv()
			team = entry[0]; fitness = entry[1][0]; frames = entry[2]

			for agent_id, popn_id in enumerate(team): self.agents[agent_id].fitnesses[popn_id].append(fitness[0]) ##Assign
			all_fits.append(fitness)
			self.total_frames+=frames


		####### JOIN PG ROLLOUTS ########
		if USE_PG and not RANDOM_BASELINE:
			entry = self.pg_result_pipes[1].recv()
			pg_fits = entry[1][0]
			self.total_frames += entry[2]


		####### JOIN TEST ROLLOUTS ########
		test_fits = []
		if gen % TEST_GAP == 0:
			entry = self.test_result_pipes[1].recv()
			test_fits = entry[1][0]
			test_tracker.update([mod.list_mean(test_fits)], self.total_frames)


		#Evolution Step
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
	torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)    #Seeds

	# INITIALIZE THE MAIN AGENT CLASS
	ai = MERL(args)
	print(' State_dim:', args.state_dim, 'Action_dim', args.action_dim)
	time_start = time.time()

	###### TRAINING LOOP ########
	for gen in range(1, args.frames_bound): #RUN VIRTUALLY FOREVER

		#ONE EPOCH OF TRAINING
		popn_fits, pg_fits, test_fits = ai.train(gen, test_tracker)


		#PRINT PROGRESS
		print('Ep:/Frames', gen, '/', ai.total_frames, 'Popn stat:', mod.list_stat(popn_fits), 'PG_stat:', mod.list_stat(pg_fits),
			  'Average:',pprint(test_tracker.all_tracker[0][1]), 'FPS:',pprint(ai.total_frames/(time.time()-time_start))
			  )

		if gen % 5 ==0:
			print()
			print('Test_stat:', mod.list_stat(test_fits))
			print('SAVETAG:  ',args.savetag)
			print()









