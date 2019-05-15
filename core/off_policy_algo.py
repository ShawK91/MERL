import torch, os
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from core import mod_utils as utils
from core.models import Actor, QNetwork, ValueNetwork, ActualizationNetwork, MultiHeadActor


class TD3(object):
	"""Classes implementing TD3 and DDPG off-policy learners

		 Parameters:
			   args (object): Parameter class


	 """
	def __init__(self, id, algo_name, state_dim, action_dim, hidden_size, actor_lr, critic_lr, gamma, tau, savetag, foldername, actualize, use_gpu, init_w = True):

		self.algo_name = algo_name; self.gamma = gamma; self.tau = tau; self.total_update = 0; self.agent_id = id;	self.actualize = actualize; self.use_gpu = use_gpu
		self.tracker = utils.Tracker(foldername, ['q_'+savetag, 'qloss_'+savetag, 'policy_loss_'+savetag, 'alz_score'+savetag,'alz_policy'+savetag], '.csv', save_iteration=1000, conv_size=1000)

		#Initialize actors
		self.policy = Actor(state_dim, action_dim, hidden_size, policy_type='DeterministicPolicy')
		if init_w: self.policy.apply(utils.init_weights)
		self.policy_target = Actor(state_dim, action_dim, hidden_size, policy_type='DeterministicPolicy')
		utils.hard_update(self.policy_target, self.policy)
		self.policy_optim = Adam(self.policy.parameters(), actor_lr)


		self.critic = QNetwork(state_dim, action_dim,hidden_size)
		if init_w: self.critic.apply(utils.init_weights)
		self.critic_target = QNetwork(state_dim, action_dim, hidden_size)
		utils.hard_update(self.critic_target, self.critic)
		self.critic_optim = Adam(self.critic.parameters(), critic_lr)

		if actualize:
			self.ANetwork = ActualizationNetwork(state_dim, action_dim, hidden_size)
			if init_w: self.ANetwork.apply(utils.init_weights)
			self.actualize_optim = Adam(self.ANetwork.parameters(), critic_lr)
			self.actualize_lr = 0.2
			if use_gpu: self.ANetwork.cuda()

		self.loss = nn.MSELoss()

		if use_gpu:
			self.policy_target.cuda(); self.critic_target.cuda(); self.policy.cuda(); self.critic.cuda()
		self.num_critic_updates = 0

		#Statistics Tracker
		#self.action_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.policy_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.alz_score = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.alz_policy = {'min':None, 'max': None, 'mean':None, 'std':None}
		#self.val = {'min':None, 'max': None, 'mean':None, 'std':None}
		#self.value_loss = {'min':None, 'max': None, 'mean':None, 'std':None}


	def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, global_reward, num_epoch=1, **kwargs):
		"""Runs a step of Bellman upodate and policy gradient using a batch of experiences

			 Parameters:
				  state_batch (tensor): Current States
				  next_state_batch (tensor): Next States
				  action_batch (tensor): Actions
				  reward_batch (tensor): Rewards
				  done_batch (tensor): Done batch
				  num_epoch (int): Number of learning iteration to run with the same data

			 Returns:
				   None

		 """

		if isinstance(state_batch, list): state_batch = torch.cat(state_batch); next_state_batch = torch.cat(next_state_batch); action_batch = torch.cat(action_batch); reward_batch = torch.cat(reward_batch). done_batch = torch.cat(done_batch); global_reward = torch.cat(global_reward)

		for _ in range(num_epoch):
			########### CRITIC UPDATE ####################

			#Compute next q-val, next_v and target
			with torch.no_grad():
				#Policy Noise
				policy_noise = np.random.normal(0, kwargs['policy_noise'], (action_batch.size()[0], action_batch.size()[1]))
				policy_noise = torch.clamp(torch.Tensor(policy_noise), -kwargs['policy_noise_clip'], kwargs['policy_noise_clip'])

				#Compute next action_bacth
				next_action_batch = self.policy_target.clean_action(next_state_batch, return_only_action=True) + policy_noise.cuda() if self.use_gpu else policy_noise
				next_action_batch = torch.clamp(next_action_batch, -1, 1)

				#Compute Q-val and value of next state masking by done
				q1, q2 = self.critic_target.forward(next_state_batch, next_action_batch)
				q1 = (1 - done_batch) * q1
				q2 = (1 - done_batch) * q2
				#next_val = (1 - done_batch) * next_val

				#Select which q to use as next-q (depends on algo)
				if self.algo_name == 'TD3' or self.algo_name == 'TD3_actor_min': next_q = torch.min(q1, q2)
				elif self.algo_name == 'DDPG': next_q = q1
				elif self.algo_name == 'TD3_max': next_q = torch.max(q1, q2)

				#Compute target q and target val
				target_q = reward_batch + (self.gamma * next_q)
				#if self.args.use_advantage: target_val = reward_batch + (self.gamma * next_val)

			if self.actualize:
				##########Actualization Network Update
				current_Ascore = self.ANetwork.forward(state_batch, action_batch)
				utils.compute_stats(current_Ascore, self.alz_score)
				target_Ascore = (self.actualize_lr) * (global_reward * 10.0) + (1 - self.actualize_lr) * current_Ascore.detach()
				actualize_loss = self.loss(target_Ascore, current_Ascore).mean()



			self.critic_optim.zero_grad()
			current_q1, current_q2 = self.critic.forward((state_batch), (action_batch))
			utils.compute_stats(current_q1, self.q)

			dt = self.loss(current_q1, target_q)
			# if self.args.use_advantage:
			#     dt = dt + self.loss(current_val, target_val)
			#     utils.compute_stats(current_val, self.val)

			if self.algo_name == 'TD3' or self.algo_name == 'TD3_max': dt = dt + self.loss(current_q2, target_q)
			utils.compute_stats(dt, self.q_loss)

			# if self.args.critic_constraint:
			#     if dt.item() > self.args.critic_constraint_w:
			#         dt = dt * (abs(self.args.critic_constraint_w / dt.item()))
			dt.backward()

			self.critic_optim.step()
			self.num_critic_updates += 1

			if self.actualize:
				self.actualize_optim.zero_grad()
				actualize_loss.backward()
				self.actualize_optim.step()


			#Delayed Actor Update
			if self.num_critic_updates % kwargs['policy_ups_freq'] == 0:

				actor_actions = self.policy.clean_action(state_batch, return_only_action=False)

				# # Trust Region constraint
				# if self.args.trust_region_actor:
				#     with torch.no_grad(): old_actor_actions = self.actor_target.forward(state_batch)
				#     actor_actions = action_batch - old_actor_actions


				Q1, Q2 = self.critic.forward(state_batch, actor_actions)

				# if self.args.use_advantage: policy_loss = -(Q1 - val)
				policy_loss = -Q1

				utils.compute_stats(-policy_loss,self.policy_loss)
				policy_loss = policy_loss.mean()

				###Actualzie Policy Update
				if self.actualize:
					A1 = self.ANetwork.forward(state_batch, actor_actions)
					utils.compute_stats(A1, self.alz_policy)
					policy_loss += -A1.mean()*0.1



				self.policy_optim.zero_grad()



				policy_loss.backward(retain_graph=True)
				#nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
				# if self.args.action_loss:
				#     action_loss = torch.abs(actor_actions-0.5)
				#     utils.compute_stats(action_loss, self.action_loss)
				#     action_loss = action_loss.mean() * self.args.action_loss_w
				#     action_loss.backward()
				#     #if self.action_loss[-1] > self.policy_loss[-1]: self.args.action_loss_w *= 0.9 #Decay action_w loss if action loss is larger than policy gradient loss
				self.policy_optim.step()


			# if self.args.hard_update:
			#     if self.num_critic_updates % self.args.hard_update_freq == 0:
			#         if self.num_critic_updates % self.args.policy_ups_freq == 0: self.hard_update(self.actor_target, self.actor)
			#         self.hard_update(self.critic_target, self.critic)


			if self.num_critic_updates % kwargs['policy_ups_freq'] == 0: utils.soft_update(self.policy_target, self.policy, self.tau)
			utils.soft_update(self.critic_target, self.critic, self.tau)

			self.total_update += 1
			if self.agent_id == 0:
				self.tracker.update([self.q['mean'], self.q_loss['mean'], self.policy_loss['mean'],self.alz_score['mean'], self.alz_policy['mean']] ,self.total_update)


class SAC(object):
	def __init__(self, id, num_inputs, action_dim, hidden_size, gamma, critic_lr, actor_lr, tau, alpha, target_update_interval, savetag, foldername, actualize, use_gpu):

		self.num_inputs = num_inputs
		self.action_space = action_dim
		self.gamma = gamma
		self.tau = 0.005
		self.alpha = 0.2
		self.policy_type = "Gaussian"
		self.target_update_interval = 1
		self.tracker = utils.Tracker(foldername, ['q_'+savetag, 'qloss_'+savetag, 'value_'+savetag, 'value_loss_'+savetag, 'policy_loss_'+savetag, 'mean_loss_'+savetag, 'std_loss_'+savetag], '.csv',save_iteration=1000, conv_size=1000)
		self.total_update = 0
		self.agent_id = id
		self.actualize = actualize

		self.critic = QNetwork(self.num_inputs, self.action_space, hidden_size)
		self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
		self.soft_q_criterion = nn.MSELoss()

		if self.policy_type == "Gaussian":
			self.policy = Actor(self.num_inputs, self.action_space, hidden_size, policy_type='GaussianPolicy')
			self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)

			self.value = ValueNetwork(self.num_inputs, hidden_size)
			self.value_target = ValueNetwork(self.num_inputs, hidden_size)
			self.value_optim = Adam(self.value.parameters(), lr=critic_lr)
			utils.hard_update(self.value_target, self.value)
			self.value_criterion = nn.MSELoss()
		else:
			self.policy = Actor(self.num_inputs, self.action_space, hidden_size, policy_type='DeterministicPolicy')
			self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)

			self.critic_target = QNetwork(self.num_inputs, self.action_space, hidden_size)
			utils.hard_update(self.critic_target, self.critic)

		self.policy.cuda()
		self.value.cuda()
		self.value_target.cuda()
		self.critic.cuda()

		#Statistics Tracker
		self.q = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.val = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.value_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.policy_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.mean_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.std_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q_loss = {'min':None, 'max': None, 'mean':None, 'std':None}



	# def select_action(self, state, eval=False):
	#     state = torch.FloatTensor(state).unsqueeze(0)
	#     if eval == False:
	#         self.policy.train()
	#         action, _, _, _, _ = self.policy.evaluate(state)
	#     else:
	#         self.policy.eval()
	#         _, _, _, action, _ = self.policy.evaluate(state)
	#
	#     # action = torch.tanh(action)
	#     action = action.detach().cpu().numpy()
	#     return action[0]

	def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, mask_batch, updates, **ignore):
		# state_batch = torch.FloatTensor(state_batch)
		# next_state_batch = torch.FloatTensor(next_state_batch)
		# action_batch = torch.FloatTensor(action_batch)
		# reward_batch = torch.FloatTensor(reward_batch)
		# mask_batch = torch.FloatTensor(np.float32(mask_batch))

		# reward_batch = reward_batch.unsqueeze(1)  # reward_batch = [batch_size, 1]
		# mask_batch = mask_batch.unsqueeze(1)  # mask_batch = [batch_size, 1]

		"""
		Use two Q-functions to mitigate positive bias in the policy improvement step that is known
		to degrade performance of value based methods. Two Q-functions also significantly speed
		up training, especially on harder task.
		"""
		expected_q1_value, expected_q2_value = self.critic(state_batch, action_batch)
		new_action, log_prob, _, mean, log_std = self.policy.noisy_action(state_batch, return_only_action=False)
		utils.compute_stats(expected_q1_value, self.q)


		if self.policy_type == "Gaussian":
			"""
			Including a separate function approximator for the soft value can stabilize training.
			"""
			expected_value = self.value(state_batch)
			utils.compute_stats(expected_value, self.val)
			target_value = self.value_target(next_state_batch)
			next_q_value = reward_batch + mask_batch * self.gamma * target_value  # Reward Scale * r(st,at) - γV(target)(st+1))
		else:
			"""
			There is no need in principle to include a separate function approximator for the state value.
			We use a target critic network for deterministic policy and eradicate the value value network completely.
			"""
			next_state_action, _, _, _, _, = self.policy.noisy_action(next_state_batch, return_only_action=False)
			target_critic_1, target_critic_2 = self.critic_target(next_state_batch, next_state_action)
			target_critic = torch.min(target_critic_1, target_critic_2)
			next_q_value = reward_batch + mask_batch * self.gamma * target_critic  # Reward Scale * r(st,at) - γQ(target)(st+1)

		"""
		Soft Q-function parameters can be trained to minimize the soft Bellman residual
		JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
		∇JQ = ∇Q(st,at)(Q(st,at) - r(st,at) - γV(target)(st+1))
		"""
		q1_value_loss = self.soft_q_criterion(expected_q1_value, next_q_value.detach())
		q2_value_loss = self.soft_q_criterion(expected_q2_value, next_q_value.detach())
		utils.compute_stats(q1_value_loss, self.q_loss)
		q1_new, q2_new = self.critic(state_batch, new_action)
		expected_new_q_value = torch.min(q1_new, q2_new)

		if self.policy_type == "Gaussian":
			"""
			Including a separate function approximator for the soft value can stabilize training and is convenient to 
			train simultaneously with the other networks
			Update the V towards the min of two Q-functions in order to reduce overestimation bias from function approximation error.
			JV = 𝔼st~D[0.5(V(st) - (𝔼at~π[Qmin(st,at) - log π(at|st)]))^2]
			∇JV = ∇V(st)(V(st) - Q(st,at) + logπ(at|st))
			"""
			next_value = expected_new_q_value - (self.alpha * log_prob)
			value_loss = self.value_criterion(expected_value, next_value.detach())
			utils.compute_stats(value_loss, self.value_loss)
		else:
			pass

		"""
		Reparameterization trick is used to get a low variance estimator
		f(εt;st) = action sampled from the policy
		εt is an input noise vector, sampled from some fixed distribution
		Jπ = 𝔼st∼D,εt∼N[logπ(f(εt;st)|st)−Q(st,f(εt;st))]
		∇Jπ =∇log π + ([∇at log π(at|st) − ∇at Q(st,at)])∇f(εt;st)
		"""
		policy_loss = ((self.alpha * log_prob) - expected_new_q_value)
		utils.compute_stats(policy_loss, self.policy_loss)
		policy_loss = policy_loss.mean()

		# Regularization Loss
		mean_loss = 0.001 * mean.pow(2)
		std_loss = 0.001 * log_std.pow(2)
		utils.compute_stats(mean_loss, self.mean_loss)
		utils.compute_stats(std_loss, self.std_loss)
		mean_loss = mean_loss.mean()
		std_loss = std_loss.mean()


		policy_loss += mean_loss + std_loss

		self.critic_optim.zero_grad()
		q1_value_loss.backward()
		self.critic_optim.step()

		self.critic_optim.zero_grad()
		q2_value_loss.backward()
		self.critic_optim.step()

		if self.policy_type == "Gaussian":
			self.value_optim.zero_grad()
			value_loss.backward()
			self.value_optim.step()
		else:
			value_loss = torch.tensor(0.)

		self.policy_optim.zero_grad()
		policy_loss.backward()
		self.policy_optim.step()

		self.total_update += 1
		if self.agent_id == 0:
			self.tracker.update([self.q['mean'], self.q_loss['mean'], self.val['mean'], self.value_loss['mean']
								, self.policy_loss['mean'], self.mean_loss['mean'], self.std_loss['mean']], self.total_update)

		"""
		We update the target weights to match the current value function weights periodically
		Update target parameter after every n(args.target_update_interval) updates
		"""
		if updates % self.target_update_interval == 0 and self.policy_type == "Deterministic":
			utils.soft_update(self.critic_target, self.critic, self.tau)

		elif updates % self.target_update_interval == 0 and self.policy_type == "Gaussian":
			utils.soft_update(self.value_target, self.value, self.tau)
		return value_loss.item(), q1_value_loss.item(), q2_value_loss.item(), policy_loss.item()

	# Save model parameters
	def save_model(self, env_name, suffix="", actor_path=None, critic_path=None, value_path=None):
		if not os.path.exists('models/'):
			os.makedirs('models/')

		if actor_path is None:
			actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
		if critic_path is None:
			critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
		if value_path is None:
			value_path = "models/sac_value_{}_{}".format(env_name, suffix)
		print('Saving models to {}, {} and {}'.format(actor_path, critic_path, value_path))
		torch.save(self.value.state_dict(), value_path)
		torch.save(self.policy.state_dict(), actor_path)
		torch.save(self.critic.state_dict(), critic_path)

	# Load model parameters
	def load_model(self, actor_path, critic_path, value_path):
		print('Loading models from {}, {} and {}'.format(actor_path, critic_path, value_path))
		if actor_path is not None:
			self.policy.load_state_dict(torch.load(actor_path))
		if critic_path is not None:
			self.critic.load_state_dict(torch.load(critic_path))
		if value_path is not None:
			self.value.load_state_dict(torch.load(value_path))


class MultiTD3(object):
	"""Classes implementing TD3 and DDPG off-policy learners

		 Parameters:
			   args (object): Parameter class


	 """
	def __init__(self, id, algo_name, state_dim, action_dim, hidden_size, actor_lr, critic_lr, gamma, tau, savetag, foldername, actualize, use_gpu, num_agents, init_w = True):

		self.algo_name = algo_name; self.gamma = gamma; self.tau = tau; self.total_update = 0; self.agent_id = id;	self.actualize = actualize; self.use_gpu = use_gpu
		self.tracker = utils.Tracker(foldername, ['q_'+savetag, 'qloss_'+savetag, 'policy_loss_'+savetag, 'alz_score'+savetag,'alz_policy'+savetag], '.csv', save_iteration=1000, conv_size=1000)

		#Initialize actors
		self.policy = MultiHeadActor(state_dim, action_dim, hidden_size, num_agents)
		if init_w: self.policy.apply(utils.init_weights)
		self.policy_target = MultiHeadActor(state_dim, action_dim, hidden_size, num_agents)
		utils.hard_update(self.policy_target, self.policy)
		self.policy_optim = Adam(self.policy.parameters(), actor_lr)


		self.critic = QNetwork(state_dim, action_dim,hidden_size)
		if init_w: self.critic.apply(utils.init_weights)
		self.critic_target = QNetwork(state_dim, action_dim, hidden_size)
		utils.hard_update(self.critic_target, self.critic)
		self.critic_optim = Adam(self.critic.parameters(), critic_lr)

		if actualize:
			self.ANetwork = ActualizationNetwork(state_dim, action_dim, hidden_size)
			if init_w: self.ANetwork.apply(utils.init_weights)
			self.actualize_optim = Adam(self.ANetwork.parameters(), critic_lr)
			self.actualize_lr = 0.2
			if use_gpu: self.ANetwork.cuda()

		self.loss = nn.MSELoss()

		if use_gpu:
			self.policy_target.cuda(); self.critic_target.cuda(); self.policy.cuda(); self.critic.cuda()
		self.num_critic_updates = 0

		#Statistics Tracker
		#self.action_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.policy_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.alz_score = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.alz_policy = {'min':None, 'max': None, 'mean':None, 'std':None}
		#self.val = {'min':None, 'max': None, 'mean':None, 'std':None}
		#self.value_loss = {'min':None, 'max': None, 'mean':None, 'std':None}


	def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, global_reward, agent_id, num_epoch=1, **kwargs):
		"""Runs a step of Bellman upodate and policy gradient using a batch of experiences

			 Parameters:
				  state_batch (tensor): Current States
				  next_state_batch (tensor): Next States
				  action_batch (tensor): Actions
				  reward_batch (tensor): Rewards
				  done_batch (tensor): Done batch
				  num_epoch (int): Number of learning iteration to run with the same data

			 Returns:
				   None

		 """

		if isinstance(state_batch, list): state_batch = torch.cat(state_batch); next_state_batch = torch.cat(next_state_batch); action_batch = torch.cat(action_batch); reward_batch = torch.cat(reward_batch). done_batch = torch.cat(done_batch); global_reward = torch.cat(global_reward)

		for _ in range(num_epoch):
			########### CRITIC UPDATE ####################

			#Compute next q-val, next_v and target
			with torch.no_grad():
				#Policy Noise
				policy_noise = np.random.normal(0, kwargs['policy_noise'], (action_batch.size()[0], action_batch.size()[1]))
				policy_noise = torch.clamp(torch.Tensor(policy_noise), -kwargs['policy_noise_clip'], kwargs['policy_noise_clip'])

				#Compute next action_bacth
				next_action_batch = self.policy_target.clean_action(next_state_batch, agent_id) + policy_noise.cuda() if self.use_gpu else policy_noise
				next_action_batch = torch.clamp(next_action_batch, -1, 1)

				#Compute Q-val and value of next state masking by done
				q1, q2 = self.critic_target.forward(next_state_batch, next_action_batch)
				q1 = (1 - done_batch) * q1
				q2 = (1 - done_batch) * q2
				#next_val = (1 - done_batch) * next_val

				#Select which q to use as next-q (depends on algo)
				if self.algo_name == 'TD3' or self.algo_name == 'TD3_actor_min': next_q = torch.min(q1, q2)
				elif self.algo_name == 'DDPG': next_q = q1
				elif self.algo_name == 'TD3_max': next_q = torch.max(q1, q2)

				#Compute target q and target val
				target_q = reward_batch + (self.gamma * next_q)
				#if self.args.use_advantage: target_val = reward_batch + (self.gamma * next_val)

			if self.actualize:
				##########Actualization Network Update
				current_Ascore = self.ANetwork.forward(state_batch, action_batch)
				utils.compute_stats(current_Ascore, self.alz_score)
				target_Ascore = (self.actualize_lr) * (global_reward * 10.0) + (1 - self.actualize_lr) * current_Ascore.detach()
				actualize_loss = self.loss(target_Ascore, current_Ascore).mean()



			self.critic_optim.zero_grad()
			current_q1, current_q2 = self.critic.forward((state_batch), (action_batch))
			utils.compute_stats(current_q1, self.q)

			dt = self.loss(current_q1, target_q)
			# if self.args.use_advantage:
			#     dt = dt + self.loss(current_val, target_val)
			#     utils.compute_stats(current_val, self.val)

			if self.algo_name == 'TD3' or self.algo_name == 'TD3_max': dt = dt + self.loss(current_q2, target_q)
			utils.compute_stats(dt, self.q_loss)

			# if self.args.critic_constraint:
			#     if dt.item() > self.args.critic_constraint_w:
			#         dt = dt * (abs(self.args.critic_constraint_w / dt.item()))
			dt.backward()

			self.critic_optim.step()
			self.num_critic_updates += 1

			if self.actualize:
				self.actualize_optim.zero_grad()
				actualize_loss.backward()
				self.actualize_optim.step()


			#Delayed Actor Update
			if self.num_critic_updates % kwargs['policy_ups_freq'] == 0:

				actor_actions = self.policy.clean_action(state_batch, agent_id)

				# # Trust Region constraint
				# if self.args.trust_region_actor:
				#     with torch.no_grad(): old_actor_actions = self.actor_target.forward(state_batch)
				#     actor_actions = action_batch - old_actor_actions


				Q1, Q2 = self.critic.forward(state_batch, actor_actions)

				# if self.args.use_advantage: policy_loss = -(Q1 - val)
				policy_loss = -Q1

				utils.compute_stats(-policy_loss,self.policy_loss)
				policy_loss = policy_loss.mean()

				###Actualzie Policy Update
				if self.actualize:
					A1 = self.ANetwork.forward(state_batch, actor_actions)
					utils.compute_stats(A1, self.alz_policy)
					policy_loss += -A1.mean()



				self.policy_optim.zero_grad()



				policy_loss.backward(retain_graph=True)
				#nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
				# if self.args.action_loss:
				#     action_loss = torch.abs(actor_actions-0.5)
				#     utils.compute_stats(action_loss, self.action_loss)
				#     action_loss = action_loss.mean() * self.args.action_loss_w
				#     action_loss.backward()
				#     #if self.action_loss[-1] > self.policy_loss[-1]: self.args.action_loss_w *= 0.9 #Decay action_w loss if action loss is larger than policy gradient loss
				self.policy_optim.step()


			# if self.args.hard_update:
			#     if self.num_critic_updates % self.args.hard_update_freq == 0:
			#         if self.num_critic_updates % self.args.policy_ups_freq == 0: self.hard_update(self.actor_target, self.actor)
			#         self.hard_update(self.critic_target, self.critic)


			if self.num_critic_updates % kwargs['policy_ups_freq'] == 0: utils.soft_update(self.policy_target, self.policy, self.tau)
			utils.soft_update(self.critic_target, self.critic, self.tau)

			self.total_update += 1
			if self.agent_id == 0:
				self.tracker.update([self.q['mean'], self.q_loss['mean'], self.policy_loss['mean'],self.alz_score['mean'], self.alz_policy['mean']] ,self.total_update)


class MADDPG_reference:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.5
        self.tau = 0.0001

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.00005) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.00005) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None, batch.next_states)))

            # state_batch: batch_size x n_agents x dim_obs
            state_batch = th.stack(batch.states).type(FloatTensor)
            action_batch = th.stack(batch.actions).type(FloatTensor)
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = [
                self.actors_target[i](non_final_next_states[:,
                                                            i,
                                                            :]) for i in range(
                                                                self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())

            target_Q = th.zeros(
                self.batch_size).type(FloatTensor)

            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1,
                                            self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions

            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        # if self.steps_done % 100 == 0 and self.steps_done > 0:
        for i in range(self.n_agents):
            soft_update(self.critics_target[i], self.critics[i], self.tau)
            soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch, noise=True):
        # state_batch: n_agents x state_dim
        actions = th.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i](sb)#(sb.unsqueeze(0)).squeeze()

            if noise:
                act += th.from_numpy(np.random.randn(2) * self.var[i]).type(FloatTensor)

                if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                    self.var[i] *= 0.999998
                act = th.clamp(act, -1.0, 1.0)

            actions[i, :] = act
        self.steps_done += 1

        return actions


class MATD3(object):
	"""Classes implementing TD3 and DDPG off-policy learners

		 Parameters:
			   args (object): Parameter class


	 """
	def __init__(self, id, algo_name, state_dim, action_dim, hidden_size, actor_lr, critic_lr, gamma, tau, savetag, foldername, actualize, use_gpu, num_agents, init_w = True):

		self.algo_name = algo_name; self.gamma = gamma; self.tau = tau; self.total_update = 0; self.agent_id = id;self.use_gpu = use_gpu
		self.tracker = utils.Tracker(foldername, ['q_'+savetag, 'qloss_'+savetag, 'policy_loss_'+savetag], '.csv', save_iteration=1000, conv_size=1000)
		self.num_agents = num_agents

		#Initialize actors
		self.policy = MultiHeadActor(state_dim, action_dim, hidden_size, num_agents)
		if init_w: self.policy.apply(utils.init_weights)
		self.policy_target = MultiHeadActor(state_dim, action_dim, hidden_size, num_agents)
		utils.hard_update(self.policy_target, self.policy)
		self.policy_optim = Adam(self.policy.parameters(), actor_lr)


		self.critics = [QNetwork(state_dim*num_agents, action_dim*num_agents, hidden_size*3) for _ in range(num_agents)]

		self.critics_target = [QNetwork(state_dim*num_agents, action_dim*num_agents, hidden_size*3) for _ in range(num_agents)]
		if init_w:
			for critic, critic_target in zip(self.critics, self.critics_target):
				critic.apply(utils.init_weights)
				utils.hard_update(critic_target, critic)
		self.critic_optims = [Adam(critic.parameters(), critic_lr) for critic in self.critics]


		self.loss = nn.MSELoss()

		if use_gpu:
			self.policy_target.cuda(); self.policy.cuda()
			for critic, critic_target in zip(self.critics, self.critics_target):
				critic.cuda()
				critic_target.cuda()


		self.num_critic_updates = 0

		#Statistics Tracker
		#self.action_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.policy_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
		self.q = {'min':None, 'max': None, 'mean':None, 'std':None}



	def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, global_reward, agent_id, num_epoch=1, **kwargs):
		"""Runs a step of Bellman upodate and policy gradient using a batch of experiences

			 Parameters:
				  state_batch (tensor): Current States
				  next_state_batch (tensor): Next States
				  action_batch (tensor): Actions
				  reward_batch (tensor): Rewards
				  done_batch (tensor): Done batch
				  num_epoch (int): Number of learning iteration to run with the same data

			 Returns:
				   None

		 """

		if isinstance(state_batch, list): state_batch = torch.cat(state_batch); next_state_batch = torch.cat(next_state_batch); action_batch = torch.cat(action_batch); reward_batch = torch.cat(reward_batch). done_batch = torch.cat(done_batch); global_reward = torch.cat(global_reward)
		batch_size = len(state_batch)

		for _ in range(num_epoch):
			########### CRITIC UPDATE ####################

			#Compute next q-val, next_v and target
			with torch.no_grad():


				#Compute next action_bacth
				next_action_batch = torch.cat([self.policy_target.clean_action(next_state_batch[:, id, :], id) for id in range(self.num_agents)], 1)
				if self.algo_name == 'TD3':
					# Policy Noise
					policy_noise = np.random.normal(0, kwargs['policy_noise'], (action_batch.size()[0], action_batch.size()[1] * action_batch.size()[2]))
					policy_noise = torch.clamp(torch.Tensor(policy_noise), -kwargs['policy_noise_clip'], kwargs['policy_noise_clip'])
					next_action_batch += policy_noise.cuda() if self.use_gpu else policy_noise
				next_action_batch = torch.clamp(next_action_batch, -1, 1)

				#Compute Q-val and value of next state masking by done

				q1, q2 = self.critics_target[agent_id].forward(next_state_batch.view(batch_size, -1), next_action_batch)
				q1 = (1 - done_batch) * q1
				q2 = (1 - done_batch) * q2
				#next_val = (1 - done_batch) * next_val

				#Select which q to use as next-q (depends on algo)
				if self.algo_name == 'TD3':next_q = torch.min(q1, q2)
				elif self.algo_name == 'DDPG': next_q = q1

				#Compute target q and target val
				target_q = reward_batch[:,agent_id].unsqueeze(1) + (self.gamma * next_q)
				#if self.args.use_advantage: target_val = reward_batch + (self.gamma * next_val)



			self.critic_optims[agent_id].zero_grad()
			current_q1, current_q2 = self.critics[agent_id].forward((state_batch.view(batch_size, -1)), (action_batch.view(batch_size, -1)))
			utils.compute_stats(current_q1, self.q)

			dt = self.loss(current_q1, target_q)
			# if self.args.use_advantage:
			#     dt = dt + self.loss(current_val, target_val)
			#     utils.compute_stats(current_val, self.val)

			if self.algo_name == 'TD3': dt = dt + self.loss(current_q2, target_q)
			utils.compute_stats(dt, self.q_loss)

			# if self.args.critic_constraint:
			#     if dt.item() > self.args.critic_constraint_w:
			#         dt = dt * (abs(self.args.critic_constraint_w / dt.item()))
			dt.backward()

			self.critic_optims[agent_id].step()
			self.num_critic_updates += 1

			#Delayed Actor Update
			if self.num_critic_updates % kwargs['policy_ups_freq'] == 0 or self.algo_name == 'DDPG':

				agent_action = self.policy.clean_action(state_batch[:,agent_id,:], agent_id)
				joint_action = action_batch.clone()
				joint_action[:,agent_id,:] = agent_action[:]

				#print(np.max(torch.abs(joint_action - action_batch).detach().cpu().numpy()), np.max(torch.abs(joint_action[:,agent_id,:] - agent_action).detach().cpu().numpy()))
				# # Trust Region constraint
				# if self.args.trust_region_actor:
				#     with torch.no_grad(): old_actor_actions = self.actor_target.forward(state_batch)
				#     actor_actions = action_batch - old_actor_actions


				Q1, Q2 = self.critics[agent_id].forward(state_batch.view(batch_size, -1), joint_action.view(batch_size, -1))

				# if self.args.use_advantage: policy_loss = -(Q1 - val)
				policy_loss = -Q1

				utils.compute_stats(-policy_loss,self.policy_loss)
				policy_loss = policy_loss.mean()


				self.policy_optim.zero_grad()



				policy_loss.backward(retain_graph=True)
				#nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
				# if self.args.action_loss:
				#     action_loss = torch.abs(actor_actions-0.5)
				#     utils.compute_stats(action_loss, self.action_loss)
				#     action_loss = action_loss.mean() * self.args.action_loss_w
				#     action_loss.backward()
				#     #if self.action_loss[-1] > self.policy_loss[-1]: self.args.action_loss_w *= 0.9 #Decay action_w loss if action loss is larger than policy gradient loss
				self.policy_optim.step()


			# if self.args.hard_update:
			#     if self.num_critic_updates % self.args.hard_update_freq == 0:
			#         if self.num_critic_updates % self.args.policy_ups_freq == 0: self.hard_update(self.actor_target, self.actor)
			#         self.hard_update(self.critic_target, self.critic)


			if self.num_critic_updates % kwargs['policy_ups_freq'] == 0 or self.algo_name == 'DDPG': utils.soft_update(self.policy_target, self.policy, self.tau)
			for critic, critic_target in zip(self.critics, self.critics_target):
				utils.soft_update(critic_target, critic, self.tau)

			self.total_update += 1
			if self.agent_id == 0:
				self.tracker.update([self.q['mean'], self.q_loss['mean'], self.policy_loss['mean']] ,self.total_update)
