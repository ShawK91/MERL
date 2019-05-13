import numpy as np
import random
import torch
from torch.multiprocessing import Manager
from core.mod_utils import compute_stats


class Buffer():
	"""Cyclic Buffer stores experience tuples from the rollouts
		Parameters:
			capacity (int): Maximum number of experiences to hold in cyclic buffer
		"""

	def __init__(self, capacity, buffer_gpu, filter_c=None):
		self.capacity = capacity; self.buffer_gpu = buffer_gpu; self.filter_c = filter_c
		self.manager = Manager()
		self.tuples = self.manager.list() #Temporary shared buffer to get experiences from processes
		self.s = []; self.ns = []; self.a = []; self.r = []; self.done = []; self.global_reward = []

		# Temporary tensors that cane be loaded in GPU for fast sampling during gradient updates (updated each gen) --> Faster sampling - no need to cycle experiences in and out of gpu 1000 times
		self.sT = None; self.nsT = None; self.aT = None; self.rT = None; self.doneT = None; self.global_rewardT = None

		self.pg_frames = 0; self.total_frames = 0

		#Priority indices
		self.top_r = None
		self.top_g = None

		#Stats
		self.rstats = {'min': None, 'max': None, 'mean': None, 'std': None}
		self.gstats = {'min': None, 'max': None, 'mean': None, 'std': None}



	def data_filter(self, exp):

		# # #Initialize to not save
		# save_data = False
		# #
		# if self.gstats['mean'] == None or exp[6] == 'pg': save_data=True #save automatically if [gstats is unknown] or Policy Gradient
		#
		# elif self.filter_c == -1: save_data=True
		#
		# else:
		# 	prob_mass = (exp[5] - self.gstats['min']) / (self.gstats['max']-self.gstats['min']) #Normalization
		# 	prob = prob_mass.item() * self.filter_c #Coefficient
		# 	if random.random() < prob: save_data = True
		#
		# if save_data:
		self.s.append(exp[0])
		self.ns.append(exp[1])
		self.a.append(exp[2])
		self.r.append(exp[3])
		self.done.append(exp[4])
		self.global_reward.append(exp[5])
		self.pg_frames += 1
		self.total_frames += 1


	def referesh(self):
		"""Housekeeping
			Parameters:
				None
			Returns:
				None
		"""

		# Add ALL EXPERIENCE COLLECTED TO MEMORY concurrently
		for _ in range(len(self.tuples)):
			exp = self.tuples.pop()
			self.data_filter(exp)


		#Trim to make the buffer size < capacity
		while self.__len__() > self.capacity:
			self.s.pop(0); self.ns.pop(0); self.a.pop(0); self.r.pop(0); self.done.pop(0); self.global_reward.pop(0)


	def __len__(self):
		return len(self.s)

	def sample(self, batch_size, pr_rew=0.0, pr_global=0.0 ):
		"""Sample a batch of experiences from memory with uniform probability
			   Parameters:
				   batch_size (int): Size of the batch to sample
			   Returns:
				   Experience (tuple): A tuple of (state, next_state, action, shaped_reward, done) each as a numpy array with shape (batch_size, :)
		   """
		#Uniform sampling
		ind = random.sample(range(len(self.sT)), batch_size)

		if pr_global != 0.0 or pr_rew !=0.0:
			#Prioritization
			num_r = int(pr_rew * batch_size); num_global = int(pr_global * batch_size)
			ind_r = random.sample(self.top_r, num_r)
			ind_global = random.sample(self.top_g, num_global)

			ind = ind[num_r+num_global:] + ind_r + ind_global


		return self.sT[ind], self.nsT[ind], self.aT[ind], self.rT[ind], self.doneT[ind], self.global_rewardT[ind]
		#return np.vstack([self.s[i] for i in ind]), np.vstack([self.ns[i] for i in ind]), np.vstack([self.a[i] for i in ind]), np.vstack([self.r[i] for i in ind]), np.vstack([self.done[i] for i in ind])


	def tensorify(self):
		"""Method to save experiences to drive
			   Parameters:
				   None
			   Returns:
				   None
		   """
		self.referesh() #Referesh first

		if self.__len__() >1:

			self.sT = torch.tensor(np.vstack(self.s))
			self.nsT = torch.tensor(np.vstack(self.ns))
			self.aT = torch.tensor(np.vstack(self.a))
			self.rT = torch.tensor(np.vstack(self.r))
			self.doneT = torch.tensor(np.vstack(self.done))
			self.global_rewardT = torch.tensor(np.vstack(self.global_reward))
			if self.buffer_gpu:
				self.sT = self.sT.cuda()
				self.nsT = self.nsT.cuda()
				self.aT = self.aT.cuda()
				self.rT = self.rT.cuda()
				self.doneT = self.doneT.cuda()
				self.global_rewardT = self.global_rewardT.cuda()

			#Prioritized indices update
			self.top_r = list(np.argsort(np.vstack(self.r), axis=0)[-int(len(self.s)/10):])
			self.top_g = list(np.argsort(np.vstack(self.global_reward), axis=0)[-int(len(self.s) / 10):])

			#Update Stats
			compute_stats(self.rT, self.rstats)
			compute_stats(self.global_rewardT, self.gstats)

