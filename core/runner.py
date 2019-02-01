from core.env_wrapper import RoverDomainCython, RoverDomainPython
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

    env = RoverDomainCython(args)
    np.random.seed(id); random.seed(id)

    while True:
        teams_blueprint = task_pipe.recv() #Wait until a signal is received  to start rollout


        # Get the current team actors
        team = [models_bucket[agent_id][popn_id] for agent_id, popn_id in enumerate(teams_blueprint)]


        fitness = 0.0
        joint_state = env.reset(); rollout_trajectory = [[] for _ in range(args.num_agents)]
        joint_state = utils.to_tensor(np.array(joint_state))
        while True: #unless done

            joint_action = [team[i].forward(joint_state[i,:]).detach().numpy() for i in range(args.num_agents)]
            if type == 'pg':
                for action in joint_action: action += np.random.normal(0, 0.3, size=args.action_dim).clip(-1, 1)


            next_state, reward, done, info = env.step(np.array(joint_action, dtype = 'double'))  # Simulate one step in environment


            next_state = utils.to_tensor(np.array(next_state))
            fitness += sum(reward)/args.coupling

            #Push experiences to memory
            if store_transitions:
                for i in range(args.num_agents):
                    rollout_trajectory[i].append([np.expand_dims(utils.to_numpy(joint_state)[i,:], 0), np.expand_dims(utils.to_numpy(next_state)[i, :], 0),
                                                  np.expand_dims(np.array(joint_action)[i,:], 0), np.expand_dims(np.array([reward[i]], dtype="float32"), 0),
                                                  np.expand_dims(np.array([done], dtype="float32"), 0)])

            joint_state = next_state

            #DONE FLAG IS Received
            if done:
                if random.random() < 0.0:
                    env.render()

                #Push experiences to main
                for agent_id, buffer in enumerate(data_bucket):
                    for entry in rollout_trajectory[agent_id]: buffer.append(entry)

                break

        #Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([teams_blueprint, [fitness]])




