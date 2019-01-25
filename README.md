Multiagent Evolutionary Reinforcement Learning

#################################
          Code labels
#################################

main.py: Neureovolution learner that generates data --> Data Storage and bootstraps off policies from policy storage

core/runner.py: Rollout worker

core/ucb.py: Upper Confidence Bound implemented for learner selection by the meta-learner. UCB scores computed for each learner which are then used in a roulette wheel selection iteratively to fill out the resource allocation.

core/portfolio.py: Portfolio of learners which can vary in their core algo and hyperparameters

core/learner.py: Learner agent encapsulating the algo and sum-statistics

core/buffer.py: Cyclic Replay buffer

core/action_noise: Implements Ornstein–Uhlenbeck process for generating temporally correlated noise

core/env_wrapper.py: Wrapper around the Mujoco env

core/models.py: Actor model

core/neuroevolution.py: Implements Sub-Structured Based Neuroevolution (SSNE) with a dynamic population

core/off_policy_algo.py: Implements the off_policy_gradient learner (TD3/DDPG) with/or without Advantage functions, Trust Regions and HER

core/mod_utils.py: Helper functions



######################################
         Auxiliary scripts:
######################################


