import random, sys
import numpy as np
import math
import core.mod_utils as utils



class SSNE:
	"""Neuroevolution object that contains all then method to run SUb-structure based Neuroevolution (SSNE)

		Parameters:
			  args (object): parameter class


	"""

	def __init__(self, args):

		self.args=args
		self.gen = 0

		#Import Params
		self.env = args.config.env_choice
		self.popn_size = args.popn_size
		self.crossover_prob = args.crossover_prob
		self.mutation_prob = args.mutation_prob
		self.extinction_prob = args.extinction_prob  # Probability of extinction event
		self.extinction_magnituide = args.extinction_magnitude  # Probabilty of extinction for each genome, given an extinction event
		self.weight_clamp = args.weight_clamp
		self.mut_distribution = args.mut_distribution
		self.lineage_depth = args.lineage_depth
		self.ccea_reduction = args.ccea_reduction
		self.num_anchors = args.num_anchors
		self.num_elites = args.num_elites
		self.num_blends = args.num_blends
		self.scheme = args.scheme

		#RL TRACKERS
		self.rl_sync_pool = []; self.all_offs = []; self.rl_res = {"elites":0.0, 'selects': 0.0, 'discarded':0.0}; self.num_rl_syncs = 0.0001

		#Lineage scores
		self.lineage = [[] for _ in range(self.popn_size)]




	def selection_tournament(self, index_rank, num_offsprings, tournament_size):
		"""Conduct tournament selection

			Parameters:
				  index_rank (list): Ranking encoded as net_indexes
				  num_offsprings (int): Number of offsprings to generate
				  tournament_size (int): Size of tournament

			Returns:
				  offsprings (list): List of offsprings returned as a list of net indices

		"""


		total_choices = len(index_rank)
		offsprings = []
		for i in range(num_offsprings):
			winner = np.min(np.random.randint(total_choices, size=tournament_size))
			offsprings.append(index_rank[winner])

		offsprings = list(set(offsprings))  # Find unique offsprings
		if len(offsprings) % 2 != 0:  # Number of offsprings should be even
			offsprings.append(index_rank[winner])
		return offsprings

	def list_argsort(self, seq):
		"""Sort the list

			Parameters:
				  seq (list): list

			Returns:
				  sorted list

		"""
		return sorted(range(len(seq)), key=seq.__getitem__)

	def regularize_weight(self, weight, mag):
		"""Clamps on the weight magnitude (reguralizer)

			Parameters:
				  weight (float): weight
				  mag (float): max/min value for weight

			Returns:
				  weight (float): clamped weight

		"""
		if weight > mag: weight = mag
		if weight < -mag: weight = -mag
		return weight

	def crossover_inplace(self, gene1, gene2):
		"""Conduct one point crossover in place

			Parameters:
				  gene1 (object): A pytorch model
				  gene2 (object): A pytorch model

			Returns:
				None

		"""


		keys1 =  list(gene1.state_dict())
		keys2 = list(gene2.state_dict())

		for key in keys1:
			if key not in keys2: continue

			# References to the variable tensors
			W1 = gene1.state_dict()[key]
			W2 = gene2.state_dict()[key]

			if len(W1.shape) == 2: #Weights no bias
				num_variables = W1.shape[0]
				# Crossover opertation [Indexed by row]
				try: num_cross_overs = random.randint(0, int(num_variables * 0.3))  # Number of Cross overs
				except: num_cross_overs = 1
				for i in range(num_cross_overs):
					receiver_choice = random.random()  # Choose which gene to receive the perturbation
					if receiver_choice < 0.5:
						ind_cr = random.randint(0, W1.shape[0]-1)  #
						W1[ind_cr, :] = W2[ind_cr, :]
					else:
						ind_cr = random.randint(0, W1.shape[0]-1)  #
						W2[ind_cr, :] = W1[ind_cr, :]

			elif len(W1.shape) == 1: #Bias or LayerNorm
				if random.random() <0.8: continue #Crossover here with low frequency
				num_variables = W1.shape[0]
				# Crossover opertation [Indexed by row]
				#num_cross_overs = random.randint(0, int(num_variables * 0.05))  # Crossover number
				for i in range(1):
					receiver_choice = random.random()  # Choose which gene to receive the perturbation
					if receiver_choice < 0.5:
						ind_cr = random.randint(0, W1.shape[0]-1)  #
						W1[ind_cr] = W2[ind_cr]
					else:
						ind_cr = random.randint(0, W1.shape[0]-1)  #
						W2[ind_cr] = W1[ind_cr]

	def mutate_inplace(self, gene):
		"""Conduct mutation in place

			Parameters:
				  gene (object): A pytorch model

			Returns:
				None

		"""
		mut_strength = 0.1
		num_mutation_frac = 0.05
		super_mut_strength = 10
		super_mut_prob = 0.05
		reset_prob = super_mut_prob + 0.02

		num_params = len(list(gene.parameters()))
		ssne_probabilities = np.random.uniform(0, 1, num_params) * 2

		for i, param in enumerate(gene.parameters()):  # Mutate each param

			# References to the variable keys
			W = param.data
			if len(W.shape) == 2:  # Weights, no bias

				num_weights = W.shape[0] * W.shape[1]
				ssne_prob = ssne_probabilities[i]

				if random.random() < ssne_prob:
					num_mutations = random.randint(0,
						int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
					for _ in range(num_mutations):
						ind_dim1 = random.randint(0, W.shape[0]-1)
						ind_dim2 = random.randint(0, W.shape[-1]-1)
						random_num = random.random()

						if random_num < super_mut_prob:  # Super Mutation probability
							W[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[ind_dim1, ind_dim2])
						elif random_num < reset_prob:  # Reset probability
							W[ind_dim1, ind_dim2] = random.gauss(0, 0.1)
						else:  # mutauion even normal
							W[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * W[ind_dim1, ind_dim2])

						# Regularization hard limit
						W[ind_dim1, ind_dim2] = self.regularize_weight(W[ind_dim1, ind_dim2],
																	   self.weight_clamp)

			elif len(W.shape) == 1:  # Bias or layernorm
				num_weights = W.shape[0]
				ssne_prob = ssne_probabilities[i]*0.04 #Low probability of mutation here

				if random.random() < ssne_prob:
					num_mutations = random.randint(0,
						int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
					for _ in range(num_mutations):
						ind_dim = random.randint(0, W.shape[0]-1)
						random_num = random.random()

						if random_num < super_mut_prob:  # Super Mutation probability
							W[ind_dim] += random.gauss(0, super_mut_strength * W[ind_dim])
						elif random_num < reset_prob:  # Reset probability
							W[ind_dim] = random.gauss(0, 1)
						else:  # mutauion even normal
							W[ind_dim] += random.gauss(0, mut_strength * W[ind_dim])

						# Regularization hard limit
						W[ind_dim] = self.regularize_weight(W[ind_dim], self.weight_clamp)

	def reset_genome(self, gene):
		"""Reset a model's weights in place

			Parameters:
				  gene (object): A pytorch model

			Returns:
				None

		"""
		for param in (gene.parameters()):
			param.data.copy_(param.data)

	def get_anchors(self, states, pop, net_inds, lineage_rank):

		#return lineage_rank[0:self.num_anchors]

		#Compute all actions
		if self.args.ps != 'trunk' and (self.env == "rover_loose" or self.env == "rover_tight"): 		#We ignore the magnitude part (first entry in a 2-dim action vector) of the action and only measure diversity in the bearing
			actions = [pop[i].clean_action(states)[:,1] for i in net_inds]
		else:
			actions = [pop[i].clean_action(states) for i in net_inds]

		#Compute div_scores
		div_matrix = np.zeros((len(net_inds), len(net_inds)))-1
		for i in range(len(net_inds)):
			for j in range(len(net_inds)):
				if div_matrix[j,i] != -1: #Optimization for a symmetric matrix about its diagonal
					div_matrix[i,j] = div_matrix[j,i]
					continue
				div_matrix[i,j] = ((actions[i]-actions[j])**2).mean().item()

		#Get the anchor indices [indices to net_inds]
		anchor_inds = [lineage_rank[0]] #Initialize with the best one from lineage rank as the first anchor
		for _ in range(self.num_anchors-1):

			#Compute div_distance with existing anchors
			div_dist = div_matrix[lineage_rank[0]]
			for ind in anchor_inds:
				if ind == lineage_rank[0]: continue
				div_dist += div_matrix[ind]

			#Get div_rank based on the div_dist with existing probes
			div_rank = np.flip(np.argsort(div_dist))

			#Hybridize neg_scores
			neg_scores = [0 for _ in range(len(net_inds))]
			for i, div_ind in enumerate(div_rank):
				neg_scores[div_ind] += i

			for i, lineage_ind in enumerate(list(lineage_rank)):
				neg_scores[lineage_ind] += i

			#Compute hybrid rank
			hybrid_rank = self.list_argsort(neg_scores)

			#Add anchor
			continue_flag = True
			while continue_flag:
				for ind in hybrid_rank:
					if ind in anchor_inds: continue
					else:
						anchor_inds.append(ind)
						continue_flag = False
						break

		return anchor_inds

	def roulette_wheel(self, probs, num_samples):
		"""Roulette_wheel selection from a prob. distribution
	        Parameters:
	            probs (list): Probability distribution
				num_samples (int): Num of iterations to sample from distribution
	        Returns:
	            out (list): List of samples based on incoming distribution
		"""

		# Normalize
		probs = [prob - min(probs) + abs(min(probs)) for prob in probs]  # Biased translation (to positive axis) to ensure the lowest does not end up with a probability of zero
		total_prob = sum(probs) if sum(probs) != 0 else 1.0
		probs = [prob / total_prob for prob in probs]

		# Selection
		out = []
		for _ in range(num_samples):
			rand = random.random()

			for i in range(len(probs)):
				if rand < sum(probs[0:i + 1]):
					out.append(i)
					break

		# print('UCB_prob_mass', ["%.2f" % i for i in probs])
		# print('Allocation', out)
		# print()


		return out


	def evolve(self, pop, net_inds, fitness_evals, migration, states):
		"""Method to implement a round of selection and mutation operation

			Parameters:
				  pop (shared_list): Population of models
				  net_inds (list): Indices of individuals evaluated this generation
				  fitness_evals (list of lists): Fitness values for evaluated individuals
				  **migration (object): Policies from learners to be synced into population

			Returns:
				None

		"""

		self.gen+= 1


		#Convert the list of fitness values corresponding to each individual into a float [CCEA Reduction]
		if isinstance(fitness_evals[0], list):
			for i in range(len(fitness_evals)):
				if self.ccea_reduction == "mean": fitness_evals[i] = sum(fitness_evals[i])/len(fitness_evals[i])
				elif self.ccea_reduction == "leniency":fitness_evals[i] = max(fitness_evals[i])
				elif self.ccea_reduction == "min": fitness_evals[i] = min(fitness_evals[i])
				else: sys.exit('Incorrect CCEA Reduction scheme')


		#Append new fitness to lineage
		lineage_scores = [] #Tracks the average lineage score fot the generation
		for ind, fitness in zip(net_inds, fitness_evals):
			self.lineage[ind].append(fitness)
			lineage_scores.append( 0.75 * sum(self.lineage[ind])/len(self.lineage[ind]) + 0.25 * fitness) #Current fitness is weighted higher than lineage info
			if len(self.lineage[ind]) > self.lineage_depth: self.lineage[ind].pop(0) #Housekeeping


		# Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
		index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
		elitist_index = index_rank[:self.num_elites]  # Elitist indexes safeguard

		#Lineage rankings to elitists
		lineage_rank = self.list_argsort(lineage_scores[:]); lineage_rank.reverse()
		elitist_index = elitist_index + lineage_rank[:int(self.num_elites)]

		#Take out copies in elitist indices
		elitist_index = list(set(elitist_index))



		#################### MULTI_POINT SEARCH WITH ANCHORS/PROBES/BLENDS AND EXPLICIT DIVERSITY-BASED SEPARATION
		if self.scheme == 'multipoint':

			#Compute anchors
			anchor_inds = self.get_anchors(states, pop, net_inds[:], np.array(lineage_rank[:]))

			#Remove duplicates between anchors and elitists
			for i, elite in enumerate(elitist_index):
				if elite in anchor_inds: elitist_index.pop(i)

			##################### TRANSFER INDICES BACK TO POP INDICES: Change from ind in net_inds to ind referring to the real ind in pop ###############################
			elites = [net_inds[i] for i in elitist_index]
			anchors = [net_inds[i] for i in anchor_inds]
			anchor_fitnesses = [fitness_evals[i] for i in anchor_inds]
			anchor_index_ranks = [index_rank.index(i) for i in anchor_inds]
			#######################################################################################################################################################

			#Unselects are the individuals left in the population
			unselects = [ind for ind in net_inds if ind not in elites and ind not in anchors]

			#Inheritance step (sync learners to population)
			for policy in migration:
				replacee = unselects.pop(0)
				utils.hard_update(target=pop[replacee], source=policy)
				#wwid = genealogy.asexual(int(policy.wwid.item()))
				#pop[replacee].wwid[0] = wwid
				self.lineage[replacee] = [] #Reinitialize as empty

			#Sample anchors from a probability distribution formed of their relative fitnesses using a roulette wheel
			probe_allocation_inds = self.roulette_wheel(anchor_fitnesses, len(unselects)-self.num_blends)
			sampled_anchors = [anchors[i] for i in probe_allocation_inds]

			#Mutate the anchors to form probes
			for anchor_ind in sampled_anchors:
				# Mutate to form probes from anchors
				replacee = unselects.pop(0)
				utils.hard_update(target=pop[replacee], source=pop[anchor_ind])
				self.lineage[replacee] = [utils.list_mean(self.lineage[anchor_ind])]  #Inherit lineage from replacee
				self.mutate_inplace(pop[replacee])
				#genealogy.mutation(int(pop[replacee].wwid.item()), gen)

			if random.random() < 0.1:
				print('Evo_Info #Anchors', len(anchors), '#Probes_allocation', [sampled_anchors.count(i) for i in anchors], '#elites', len(elites), '#Blends', len(unselects), '#Migration', len(migration), 'Nets', len(net_inds), 'Anchor fitness Ranks', anchor_index_ranks)

			###### Create the blends to fill the rest of the unselects by crossovers #########
			# Number of unselects left should be even
			if len(unselects) % 2 != 0:
				unselects.append(unselects[random.randint(0, len(unselects)-1)])

			for i, j in zip(unselects[0::2], unselects[1::2]):
				off_i = random.choice(anchors)
				while True:
					off_j = random.choice(anchors)
					if off_j != off_i: break

				utils.hard_update(target=pop[i], source=pop[off_i])
				utils.hard_update(target=pop[j], source=pop[off_j])
				self.crossover_inplace(pop[i], pop[j])
				#wwid1 = genealogy.crossover(int(pop[off_i].wwid.item()), int(pop[off_j].wwid.item()), gen)
				#wwid2 = genealogy.crossover(int(pop[off_i].wwid.item()), int(pop[off_j].wwid.item()), gen)
				#pop[i].wwid[0] = wwid1; pop[j].wwid[0] = wwid2
				self.lineage[i] = [0.5*utils.list_mean(self.lineage[off_i]) + 0.5*utils.list_mean(self.lineage[off_j])]
				self.lineage[j] = [0.5*utils.list_mean(self.lineage[off_i]) + 0.5*utils.list_mean(self.lineage[off_j])]


			return anchors[0]


		####################### OLD EVOLVER WITHOUT MULTI_POINT SEARCH ###########
		elif self.scheme == 'standard':

			# Selection step
			offsprings = self.selection_tournament(index_rank,
			                                       num_offsprings=len(index_rank) - len(elitist_index) - len(
				                                       migration), tournament_size=3)

			# Transcripe ranked indexes from now on to refer to net indexes
			elitist_index = [net_inds[i] for i in elitist_index]
			offsprings = [net_inds[i] for i in offsprings]

			# Figure out unselected candidates
			unselects = []; new_elitists = []
			for i in range(len(pop)):
				if i in offsprings or i in elitist_index:
					continue
				else:
					unselects.append(i)
			random.shuffle(unselects)

			# Inheritance step (sync learners to population)
			for policy in migration:
				replacee = unselects.pop(0)
				utils.hard_update(target=pop[replacee], source=policy)
				# wwid = genealogy.asexual(int(policy.wwid.item()))
				# pop[replacee].wwid[0] = wwid
				self.lineage[replacee] = [sum(lineage_scores) / len(lineage_scores)]  # Initialize as average

			# Elitism step, assigning elite candidates to some unselects
			for i in elitist_index:
				if len(unselects) >= 1: replacee = unselects.pop(0)
				elif len(offsprings) >= 1: replacee = offsprings.pop(0)
				else: continue
				new_elitists.append(replacee)
				utils.hard_update(target=pop[replacee], source=pop[i])
				# wwid = genealogy.asexual(int(pop[i].wwid.item()))
				# pop[replacee].wwid[0] = wwid
				# genealogy.elite(wwid, gen)

				self.lineage[replacee] = self.lineage[i][:]

			# Crossover for unselected genes with 100 percent probability
			if len(unselects) % 2 != 0:  # Number of unselects left should be even
				unselects.append(unselects[random.randint(0, len(unselects) - 1)])
			for i, j in zip(unselects[0::2], unselects[1::2]):
				off_i = random.choice(new_elitists);
				off_j = random.choice(offsprings)
				utils.hard_update(target=pop[i], source=pop[off_i])
				utils.hard_update(target=pop[j], source=pop[off_j])
				self.crossover_inplace(pop[i], pop[j])
				# wwid1 = genealogy.crossover(int(pop[off_i].wwid.item()), int(pop[off_j].wwid.item()), gen)
				# wwid2 = genealogy.crossover(int(pop[off_i].wwid.item()), int(pop[off_j].wwid.item()), gen)
				# pop[i].wwid[0] = wwid1; pop[j].wwid[0] = wwid2

				self.lineage[i] = [
					0.5 * utils.list_mean(self.lineage[off_i]) + 0.5 * utils.list_mean(self.lineage[off_j])]
				self.lineage[j] = [
					0.5 * utils.list_mean(self.lineage[off_i]) + 0.5 * utils.list_mean(self.lineage[off_j])]

			# Crossover for selected offsprings
			for i, j in zip(offsprings[0::2], offsprings[1::2]):
				if random.random() < self.crossover_prob:
					self.crossover_inplace(pop[i], pop[j])
					# wwid1 = genealogy.crossover(int(pop[i].wwid.item()), int(pop[j].wwid.item()), gen)
					# wwid2 = genealogy.crossover(int(pop[i].wwid.item()), int(pop[j].wwid.item()), gen)
					# pop[i].wwid[0] = wwid1; pop[j].wwid[0] = wwid2
					self.lineage[i] = [
						0.5 * utils.list_mean(self.lineage[i]) + 0.5 * utils.list_mean(self.lineage[j])]
					self.lineage[j] = [
						0.5 * utils.list_mean(self.lineage[i]) + 0.5 * utils.list_mean(self.lineage[j])]

			# Mutate all genes in the population except the new elitists
			for i in range(len(pop)):
				if i not in new_elitists:  # Spare the new elitists
					if random.random() < self.mutation_prob:
						self.mutate_inplace(pop[i])
				# genealogy.mutation(int(pop[net_i].wwid.item()), gen)

			self.all_offs[:] = offsprings[:]
			return new_elitists[0]

		else:
			sys.exit('Incorrect Evolution Scheme')









