from envs.rover_domain.env_wrapper import RoverDomainPython
from core import mod_utils as utils
import numpy as np, random


#Rollout evaluate an agent in a complete game
def rollout_worker(args, id, type, task_pipe, result_pipe, data_bucket, models_bucket, store_transitions):
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
    else: NUM_EVALS = args.num_evals

    env = RoverDomainPython(args, NUM_EVALS)
    np.random.seed(id); random.seed(id)

    while True:
        teams_blueprint = task_pipe.recv() #Wait until a signal is received  to start rollout

        # Get the current team actors
        if type == 'test':
            team = models_bucket
        else:
            team = [models_bucket[agent_id][popn_id] for agent_id, popn_id in enumerate(teams_blueprint)]


        fitness = 0.0; frame=0
        joint_state = env.reset(); rollout_trajectory = [[] for _ in range(args.num_agents)]
        joint_state = utils.to_tensor(np.array(joint_state))
        while True: #unless done


            if type == 'pg':
                joint_action = [team[i].noisy_action(joint_state[i,:]).detach().numpy() for i in range(args.num_agents)]
            else:
                joint_action = [team[i].clean_action(joint_state[i, :]).detach().numpy() for i in range(args.num_agents)]
            #JOINT ACTION [agent_id, universe_id, action]

            next_state, reward, done, info = env.step(np.array(joint_action))  # Simulate one step in environment
            #State --> [agent_id, universe_id, obs]
            #reward --> [agent_id, universe_id]
            #done --> [universe_id]
            #info --> [universe_id]



            next_state = utils.to_tensor(np.array(next_state))
            fitness += np.sum(reward)/(args.coupling*NUM_EVALS)
            #print(reward)


            #Push experiences to memory
            if store_transitions:
                for agent_id in range(args.num_agents):
                    for universe_id in range(NUM_EVALS):
                        rollout_trajectory[agent_id].append([np.expand_dims(utils.to_numpy(joint_state)[agent_id,universe_id, :], 0),
                                                      np.expand_dims(utils.to_numpy(next_state)[agent_id, universe_id, :], 0),
                                                      np.expand_dims(np.array(joint_action)[agent_id,universe_id, :], 0),
                                                      np.expand_dims(np.array([reward[agent_id, universe_id]], dtype="float32"), 0),
                                                      np.expand_dims(np.array([done[universe_id]], dtype="float32"), 0)])

            joint_state = next_state
            frame+=NUM_EVALS

            #DONE FLAG IS Received
            if done[0]:
                #Push experiences to main
                if store_transitions:
                    for agent_id, buffer in enumerate(data_bucket):
                        for entry in rollout_trajectory[agent_id]: buffer.append(entry)

                break

        #Normalize fitness to be (0,1)
        max_score = 0.0
        for i in range(args.num_poi): max_score += (i+1)
        fitness = fitness/float(max_score)

        if random.random() < 0.01:
            env.render()
            print (type, id, 'Fit of rendered', '%.2f'%fitness)

        #Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([teams_blueprint, [fitness], frame])




