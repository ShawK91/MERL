from core.off_policy_algo import TD3, SAC
from torch.multiprocessing import Manager
from core.models import Actor
from core.buffer import Buffer
from core.neuroevolution import SSNE
import core.mod_utils as mod
import random

class Agent:
	"""Learner object encapsulating a local learner

		Parameters:
		algo_name (str): Algorithm Identifier
		state_dim (int): State size
		action_dim (int): Action size
		actor_lr (float): Actor learning rate
		critic_lr (float): Critic learning rate
		gamma (float): DIscount rate
		tau (float): Target network sync generate
		init_w (bool): Use kaimling normal to initialize?
		**td3args (**kwargs): arguments for TD3 algo


	"""

	def __init__(self, args, id):
		self.args = args
		self.id = id

		###Initalize neuroevolution module###
		self.evolver = SSNE(self.args)

		########Initialize population
		self.manager = Manager()
		self.popn = self.manager.list()
		for _ in range(args.popn_size):
			if args.algo_name == 'TD3': self.popn.append(Actor(args.state_dim, args.action_dim, args.hidden_size, policy_type='DeterministicPolicy'))
			else: self.popn.append(Actor(args.state_dim, args.action_dim, args.hidden_size, policy_type='GaussianPolicy'))
			self.popn[-1].eval()

		#### INITIALIZE PG ALGO #####
		if args.algo_name == 'TD3':
			self.algo = TD3(id, args.algo_name, args.state_dim, args.action_dim, args.hidden_size, args.actor_lr, args.critic_lr, args.gamma, args.tau, args.savetag, args.aux_save, args.actualize, args.use_gpu, args.init_w)
		else:
			self.algo = SAC(id, args.state_dim, args.action_dim, args.hidden_size, args.gamma, args.critic_lr, args.actor_lr, args.tau, args.alpha, args.target_update_interval, args.savetag, args.aux_save, args.actualize, args.use_gpu)

		#### Rollout Actor is a template used for MP #####
		self.rollout_actor = self.manager.list()
		for _ in range(args.rollout_size):
			if args.algo_name == 'TD3': self.rollout_actor.append(Actor(args.state_dim, args.action_dim, args.hidden_size, policy_type='DeterministicPolicy'))
			else: self.rollout_actor.append(Actor(args.state_dim, args.action_dim, args.hidden_size, policy_type='GaussianPolicy'))

		#Initalize buffer
		self.buffer = Buffer(args.buffer_size, buffer_gpu=True)

		#Agent metrics
		self.fitnesses = [[] for _ in range(args.popn_size)]

		###Best Policy HOF####
		self.champ_ind = 0



	def update_parameters(self):
		self.buffer.referesh()
		if self.buffer.__len__() < 10 * self.args.batch_size: return ###BURN_IN_PERIOD
		self.buffer.tensorify()

		td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': -1.0, 'action_high': 1.0}

		for _ in range(int(self.args.gradperstep * self.buffer.pg_frames)):
			s, ns, a, r, done, global_reward = self.buffer.sample(self.args.batch_size, pr_rew=self.args.priority_rate, pr_global=self.args.priority_rate)
			if self.args.use_gpu:
				s = s.cuda(); ns = ns.cuda(); a = a.cuda(); r = r.cuda(); done = done.cuda(); global_reward = global_reward.cuda()
			self.algo.update_parameters(s, ns, a, r, done, global_reward, 1, **td3args)

		self.buffer.pg_frames = 0 #Reset new frame counter to 0

	def evolve(self):

		## One gen of evolution ###
		if self.args.popn_size > 1: #If not no-evo

			#Make sure that the buffer has been refereshed and tensorified
			if self.buffer.__len__() < 1000: self.buffer.tensorify()
			if random.random() < 0.1: self.buffer.tensorify()

			#Get sample of states from the buffer
			if self.buffer.__len__() < 1000: sample_size = self.buffer.__len__()
			else: sample_size = 1000

			states, _,_,_,_,_ = self.buffer.sample(sample_size, pr_rew=0.0, pr_global=0.0)

			#Net indices of nets that got evaluated this generation (meant for asynchronous evolution workloads)
			net_inds = [i for i in range(len(self.popn))] #Hack for a synchronous run

			#Evolve
			if self.args.rollout_size > 0: self.champ_ind = self.evolver.evolve(self.popn, net_inds, self.fitnesses, [self.rollout_actor[0]], states)
			else: self.champ_ind = self.evolver.evolve(self.popn, net_inds, self.fitnesses, [], states)

		#Reset fitness metrics
		self.fitnesses = [[] for _ in range(self.args.popn_size)]

	def update_rollout_actor(self):
		for actor in self.rollout_actor:
			self.algo.policy.cpu()
			mod.hard_update(actor, self.algo.policy)
			if self.args.use_gpu: self.algo.policy.cuda()



class TestAgent:
	"""Learner object encapsulating a local learner

		Parameters:
		algo_name (str): Algorithm Identifier
		state_dim (int): State size
		action_dim (int): Action size
		actor_lr (float): Actor learning rate
		critic_lr (float): Critic learning rate
		gamma (float): DIscount rate
		tau (float): Target network sync generate
		init_w (bool): Use kaimling normal to initialize?
		**td3args (**kwargs): arguments for TD3 algo


	"""
	def __init__(self, args, id):
		self.args = args
		self.id = id

		#### Rollout Actor is a template used for MP #####
		self.manager = Manager()
		self.rollout_actor = self.manager.list()
		for _ in range(args.config.num_agents):
			if args.algo_name == 'TD3':
				self.rollout_actor.append(Actor(args.state_dim, args.action_dim, args.hidden_size, policy_type='DeterministicPolicy'))
			else:
				self.rollout_actor.append(Actor(args.state_dim, args.action_dim, args.hidden_size, policy_type='GaussianPolicy'))

			if self.args.is_homogeneous: break #Only need one for homogeneous workloads


	def make_champ_team(self, agents):
		for agent_id, agent in enumerate(agents):
			if self.args.popn_size <= 1: #Testing without Evo
				agent.update_rollout_actor()
				mod.hard_update(self.rollout_actor[agent_id], agent.rollout_actor[0])
			else:
				mod.hard_update(self.rollout_actor[agent_id], agent.popn[agent.champ_ind])


