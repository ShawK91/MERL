from core import mod_utils as utils
import numpy as np, random, sys


#Rollout evaluate an agent in a complete game
def rollout_worker(args, id, type, task_pipe, result_pipe, data_bucket, models_bucket, store_transitions, random_baseline):
	"""Rollout Worker runs a simulation in the environment to generate experiences and fitness values

		Parameters:
			worker_id (int): Specific Id unique to each worker spun
			task_pipe (pipe): Receiver end of the task pipe used to receive signal to start on a task
			result_pipe (pipe): Sender end of the pipe used to report back results
			noise (object): A noise generator object
			exp_list (shared list object): A shared list object managed by a manager that is used to store experience tuples
			pop (shared list object): A shared list object managed by a manager used to store all the models (actors)
			difficulty (int): Difficulty of the task
			use_rs (bool): Use behavioral reward shaping?
			store_transition (bool): Log experiences to exp_list?

		Returns:
			None
	"""

	if type == 'test': NUM_EVALS = args.num_test
	elif type == 'pg': NUM_EVALS = args.rollout_size
	elif type == 'evo': NUM_EVALS = 10 if not args.config.env_choice == 'motivate' else 1
	else: sys.exit('Incorrect type')

	if args.config.env_choice == 'multiwalker': NUM_EVALS=1
	if args.config.env_choice == 'cassie': NUM_EVALS = 1
	if args.config.env_choice == 'hyper': NUM_EVALS = 1
	if args.config.env_choice == 'pursuit': NUM_EVALS = 10




	if args.config.env_choice == 'rover_tight' or args.config.env_choice == 'rover_loose' or args.config.env_choice == 'rover_trap':
		from envs.env_wrapper import RoverDomainPython
		env = RoverDomainPython(args, NUM_EVALS)
	elif args.config.env_choice == 'motivate':
		from envs.env_wrapper import MotivateDomain
		env = MotivateDomain(args, NUM_EVALS)
	elif args.config.env_choice == 'pursuit':
		from envs.env_wrapper import Pursuit
		env = Pursuit(args, NUM_EVALS)
	elif args.config.env_choice == 'multiwalker':
		from envs.env_wrapper import MultiWalker
		env = MultiWalker(args, NUM_EVALS)
	elif args.config.env_choice == 'cassie':
		from envs.env_wrapper import Cassie
		env = Cassie(args, NUM_EVALS)
	elif args.config.env_choice == 'hyper':
		from envs.env_wrapper import PowerPlant
		env = PowerPlant(args, NUM_EVALS)
	else: sys.exit('Incorrect env type')
	np.random.seed(id); random.seed(id)

	viz_gen = 0
	while True:
		teams_blueprint = task_pipe.recv() #Wait until a signal is received  to start rollout
		if teams_blueprint == 'TERMINATE': exit(0)  # Kill yourself

		# Get the current team actors
		if args.ps == 'full' or args.ps == 'trunk':
			if type == 'test' or type == 'pg': team = [models_bucket[0] for _ in range(args.config.num_agents)]
			elif type == "evo": team = [models_bucket[0][teams_blueprint[0]] for _ in range(args.config.num_agents)]
			else: sys.exit('Incorrect type')

		else: #Heterogeneous
			if type == 'test' or type == 'pg': team = models_bucket
			elif type == "evo": team = [models_bucket[agent_id][popn_id] for agent_id, popn_id in enumerate(teams_blueprint)]
			else: sys.exit('Incorrect type')


		if args.rollout_size == 0:
			if args.scheme == 'standard': store_transitions = False
			elif args.scheme == 'multipoint' and random.random() < 0.1 and store_transitions: store_transitions = True
		fitness = [None for _ in range(NUM_EVALS)]; frame=0
		joint_state = env.reset(); rollout_trajectory = [[] for _ in range(args.config.num_agents)]

		joint_state = utils.to_tensor(np.array(joint_state))

		while True: #unless done

			if random_baseline:
				joint_action = [np.random.random((NUM_EVALS, args.state_dim))for _ in range(args.config.num_agents)]
			elif type == 'pg':
				if  args.ps == 'trunk':
					joint_action = [team[i][0].noisy_action(joint_state[i,:], head=i).detach().numpy() for i in range(args.config.num_agents)]
				else:
					joint_action = [team[i][0].noisy_action(joint_state[i, :]).detach().numpy() for i in range(args.config.num_agents)]
			else:
				if args.ps == 'trunk':
					joint_action = [team[i].clean_action(joint_state[i, :], head=i).detach().numpy() for i in range(args.config.num_agents)]
				else:
					joint_action = [team[i].clean_action(joint_state[i, :]).detach().numpy() for i in range(args.config.num_agents)]
			#JOINT ACTION [agent_id, universe_id, action]

			#Bound Action
			joint_action = np.array(joint_action).clip(-1.0, 1.0)
			next_state, reward, done, global_reward = env.step(joint_action)  # Simulate one step in environment
			#State --> [agent_id, universe_id, obs]
			#reward --> [agent_id, universe_id]
			#done --> [universe_id]
			#info --> [universe_id]


			#if args.config.env_choice == 'motivate' and type == "test": print(['%.2f'%r for r in reward], global_reward)

			next_state = utils.to_tensor(np.array(next_state))

			#Grab global reward as fitnesses
			for i, grew in enumerate(global_reward):
				if grew != None:
					fitness[i] = grew

					#Reward Shaping
					if (args.config.env_choice == 'motivate' or args.config.env_choice == 'rover_loose' or args.config.env_choice == 'rover_tight') and type == "evo":
						if args.config.is_gsl:  # Gloabl subsumes local?
							fitness[i] += sum(env.universe[i].cumulative_local)





			#Push experiences to memory
			if store_transitions:
				if not args.is_matd3 and not args.is_maddpg: #Default
					for agent_id in range(args.config.num_agents):
						for universe_id in range(NUM_EVALS):
							if not done[universe_id]:
								rollout_trajectory[agent_id].append([np.expand_dims(utils.to_numpy(joint_state)[agent_id,universe_id, :], 0),
															  np.expand_dims(utils.to_numpy(next_state)[agent_id, universe_id, :], 0),
															  np.expand_dims(joint_action[agent_id,universe_id, :], 0),
															  np.expand_dims(np.array([reward[agent_id, universe_id]], dtype="float32"), 0),
															  np.expand_dims(np.array([done[universe_id]], dtype="float32"), 0),
															  universe_id,
								                              type])

				else: #FOR MATD3
					for universe_id in range(NUM_EVALS):
						if not done[universe_id]:
							rollout_trajectory[0].append(
								[np.expand_dims(utils.to_numpy(joint_state)[:, universe_id, :], 0),
								 np.expand_dims(utils.to_numpy(next_state)[:, universe_id, :], 0),
								 np.expand_dims(joint_action[:, universe_id, :], 0), #[batch, agent_id, :]
								 np.array([reward[:, universe_id]], dtype="float32"),
								 np.expand_dims(np.array([done[universe_id]], dtype="float32"), 0),
								 universe_id,
								 type])


			joint_state = next_state
			frame+=NUM_EVALS

			if sum(done) > 0 and sum(done) != len(done):
				k = None

			#DONE FLAG IS Received
			if sum(done)==len(done):
				#Push experiences to main
				if store_transitions:
					if args.ps == 'full': #Full setup with one replay buffer
						for heap in rollout_trajectory:
							for entry in heap:
								temp_global_reward = fitness[entry[5]]
								entry[5] = np.expand_dims(np.array([temp_global_reward], dtype="float32"), 0)
								data_bucket[0].append(entry)

					else: #Heterogeneous or Trunk
						for agent_id, buffer in enumerate(data_bucket):
							for entry in rollout_trajectory[agent_id]:
								temp_global_reward = fitness[entry[5]]
								entry[5] = np.expand_dims(np.array([temp_global_reward], dtype="float32"), 0)
								buffer.append(entry)

				break

		#print(fitness)

		#Vizualization for test sets
		if type == "test" and (args.config.env_choice == 'rover_tight' or args.config.env_choice == 'rover_loose' or args.config.env_choice == 'motivate'or args.config.env_choice == 'rover_trap'):
			env.render()
			viz_gen += 5
			#print('Test trajectory lens',[len(world.rover_path[0]) for world in env.universe])
			#print (type, id, 'Fit of rendered', ['%.2f'%f for f in fitness])
			if random.random() < 0.1:
				best_performant = fitness.index(max(fitness))
				env.universe[best_performant].viz(save=True, fname=args.aux_save+str(viz_gen)+'_'+args.savetag)


		#Send back id, fitness, total length and shaped fitness using the result pipe

		result_pipe.send([teams_blueprint, [fitness], frame])




