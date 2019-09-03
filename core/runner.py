from core import mod_utils as utils
import numpy as np, random, sys


#Rollout evaluate an agent in a complete game
def rollout_worker(args, id, type, task_pipe, result_pipe, predator_data_bucket, prey_data_bucket, predators_bucket, prey_bucket, store_transitions, random_baseline):
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
    elif type == 'evo': NUM_EVALS = 10
    else: sys.exit('Incorrect type')


    from envs.env_wrapper import SimpleTag
    env = SimpleTag(args, NUM_EVALS)
    np.random.seed(id); random.seed(id)

    while True:

        teams_blueprint = task_pipe.recv() #Wait until a signal is received  to start rollout
        if teams_blueprint == 'TERMINATE': exit(0)  # Kill yourself

        # Get the current team actors
        if type == 'test' or type == 'pg': team = [predators_bucket[0] for _ in range(args.config.num_agents)]
        elif type == "evo": team = [predators_bucket[teams_blueprint[0]] for _ in range(args.config.num_agents)]
        else: sys.exit('Incorrect type')

        if args.rollout_size == 0: store_transitions = False

        fitness = [None for _ in range(NUM_EVALS)]; frame=0
        prey_fitness = [None for _ in range(NUM_EVALS)]
        prey_state, predator_state = env.reset()
        prey_rollout_trajectory = []
        predator_rollout_trajectory = [[] for _ in range(3)]

        prey_state = utils.to_tensor(np.array(prey_state))
        predator_state = utils.to_tensor(np.array(predator_state))

        while True: #unless done
            if type == 'pg':
                prey_action = [prey_bucket[0].noisy_action(prey_state[i,:], head=i).detach().numpy() for i in range(1)]
                predator_action = [team[i].noisy_action(predator_state[i,:], head=i).detach().numpy() for i in range(3)]
            else:
                prey_action = [prey_bucket[0].clean_action(prey_state[i, :], head=i).detach().numpy() for i in range(1)]
                predator_action = [team[i].clean_action(predator_state[i, :], head=i).detach().numpy() for i in range(3)]

            #JOINT ACTION [agent_id, universe_id, action]

            #Bound Action
            prey_action = np.array(prey_action).clip(-1.0, 1.0)
            predator_action = np.array(predator_action).clip(-1.0, 1.0)

            next_pred_state, next_prey_state, pred_reward, prey_reward, done, global_reward = env.step(predator_action, prey_action)  # Simulate one step in environment
            #State --> [agent_id, universe_id, obs]
            #reward --> [agent_id, universe_id]
            #done --> [universe_id]
            #info --> [universe_id]


            #if type == "test": env.universe[0].render()

            next_pred_state = utils.to_tensor(np.array(next_pred_state))
            next_prey_state = utils.to_tensor(np.array(next_prey_state))


            #Grab global reward as fitnesses
            for i, grew in enumerate(global_reward):
                if grew[0] != None:
                    fitness[i] = grew[0]
                    prey_fitness[i] = grew[1]


            #PREDATOR
            #Push experiences to memory
            if store_transitions:
                if not args.is_matd3 and not args.is_maddpg: #Default
                    for agent_id in range(args.config.num_agents):
                        for universe_id in range(NUM_EVALS):
                            if not done:
                                predator_rollout_trajectory[agent_id].append([np.expand_dims(utils.to_numpy(predator_state)[agent_id,universe_id, :], 0),
                                                              np.expand_dims(utils.to_numpy(next_pred_state)[agent_id, universe_id, :], 0),
                                                              np.expand_dims(predator_action[agent_id,universe_id, :], 0),
                                                              np.expand_dims(np.array([pred_reward[agent_id, universe_id]], dtype="float32"), 0),
                                                              np.expand_dims(np.array([done], dtype="float32"), 0),
                                                              universe_id,
                                                              type])

                else: #FOR MATD3
                    for universe_id in range(NUM_EVALS):
                        if not done:
                            predator_rollout_trajectory[0].append(
                                [np.expand_dims(utils.to_numpy(predator_state)[:, universe_id, :], 0),
                                 np.expand_dims(utils.to_numpy(next_pred_state)[:, universe_id, :], 0),
                                 np.expand_dims(predator_action[:, universe_id, :], 0), #[batch, agent_id, :]
                                 np.array([pred_reward[:, universe_id]], dtype="float32"),
                                 np.expand_dims(np.array([done], dtype="float32"), 0),
                                 universe_id,
                                 type])

            #PREY
            for universe_id in range(NUM_EVALS):
                if not done:
                    prey_rollout_trajectory.append(
                        [np.expand_dims(utils.to_numpy(prey_state)[0, universe_id, :], 0),
                         np.expand_dims(utils.to_numpy(next_prey_state)[0, universe_id, :], 0),
                         np.expand_dims(prey_action[0, universe_id, :], 0),  # [batch, agent_id, :]
                         np.array([prey_reward[:, universe_id]], dtype="float32"),
                         np.expand_dims(np.array([done], dtype="float32"), 0),
                         universe_id,
                         type])


            predator_state = next_pred_state
            prey_state = next_prey_state
            frame+=NUM_EVALS


            #DONE FLAG IS Received
            if done:
                #Push experiences to main
                if store_transitions:
                    for agent_id, buffer in enumerate(predator_data_bucket):
                        for entry in predator_rollout_trajectory[agent_id]:
                            temp_global_reward = fitness[entry[5]]
                            entry[5] = np.expand_dims(np.array([temp_global_reward], dtype="float32"), 0)
                            buffer.append(entry)

                    #PREY
                    for buffer in prey_data_bucket:
                        for entry in prey_rollout_trajectory:
                            temp_global_reward = 0.0
                            entry[5] = np.expand_dims(np.array([temp_global_reward], dtype="float32"), 0)
                            buffer.append(entry)

                break

        #print(fitness)

        #Vizualization for test sets
       # if type == "test" and (args.config.env_choice == 'rover_tight' or args.config.env_choice == 'rover_loose' or args.config.env_choice == 'motivate'or args.config.env_choice == 'rover_trap'or args.config.config == 'simple_spread'):
            #env.render()
            #viz_gen += 5
            #print('Test trajectory lens',[len(world.rover_path[0]) for world in env.universe])
            #print (type, id, 'Fit of rendered', ['%.2f'%f for f in fitness])
            # if random.random() < 0.1:
            # 	best_performant = fitness.index(max(fitness))
            # 	env.universe[best_performant].viz(save=True, fname=args.aux_save+str(viz_gen)+'_'+args.savetag)


        #Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([teams_blueprint, [fitness, prey_fitness], frame])




