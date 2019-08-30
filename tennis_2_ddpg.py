
#%%
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
wandb.init(project="marl")

from visdom import Visdom
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from unityagents import UnityEnvironment
env = UnityEnvironment(file_name='./Tennis_Windows_x86_64/Tennis.exe', no_graphics=True)
# env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

# set hyperparameters
class Config():
    def __init__(self):
        self.state_size=  env_info.vector_observations.shape[1]
        self.action_size= brain.vector_action_space_size
        self.no_agents= 1 # I'm instantiating a 1 agent ddpg for each envorment agent #len(env_info.agents)
        self.device= device
        self.gradient_clip = 1
        self.rollout_length = 1001
        self.episode_count = 3500
        self.buffer_size = int(1e5)
        self.batch_size = 128
        self.lr_actor = 1e-4
        self.lr_critic = 1e-4
        self.discount_rate = 0.99           # discount factor
        self.tau = 1e-3    
        self.weight_decay = 0
        self.theta = 0.15
        self.sigma = 0.2
config = Config()


from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent
agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=1, config=config)
agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=0, config=config)

wandb.watch((agent1.actor_local, agent1.critic_local))

from tqdm import tqdm

# training alogrithm.
def ddpg(n_episodes=300, max_t=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    
    for i_episode in tqdm(range(1, n_episodes+1)):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(config.no_agents)
        agent1.reset()
        agent2.reset()
        

        for t in range(max_t):
            actions1 = agent1.act(np.expand_dims(states[0],0))
            actions2 = agent2.act(np.expand_dims(states[1],0))
            actions = np.hstack([actions1, actions2])
            env_info = env.step(actions)[brain_name]
            
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            rewards1, rewards2  = rewards
            dones = env_info.local_done            

            ## letting the two agents share state observations,  and next_states!

            agent1.step(np.expand_dims(states[0],0), actions1, np.expand_dims(rewards1,0), np.expand_dims(next_states[0],0), np.expand_dims(dones[0],0))
            agent2.step(np.expand_dims(states[1],0), actions2, np.expand_dims(rewards2,0), np.expand_dims(next_states[1],0), np.expand_dims(dones[1],0))
            states = next_states
            score += np.max(rewards)  # max over both agents. 
            
            if np.any(dones):
                break 
                
        
        scores_deque.append(score.mean())
        scores.append(score.mean())

        wandb.log({
            'actor_loss': agent1.actor_loss, 
            'critic_loss': agent1.critic_loss,
            'score_mean': np.mean(scores_deque)})
        # wandb.log({"outputs": wandb.Histogram(tensor_of_activations)})

        if i_episode % 50 == 0:
            torch.save(agent1.actor_local.state_dict(), 'checkpoint_actor1.pth')
            torch.save(agent1.critic_local.state_dict(), 'checkpoint_critic1.pth')
            torch.save(agent2.actor_local.state_dict(), 'checkpoint_actor2.pth')
            torch.save(agent2.critic_local.state_dict(), 'checkpoint_critic2.pth')
            
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
    return scores

# try:  # load saved weights if present
    # agent1.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    # agent1.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
# except:
#     pass


def play_round(env, brain_name, policy1, policy2, config):
    env_info = env.reset(train_mode=False)[brain_name]    
    states = env_info.vector_observations                 
    scores = np.zeros(config.no_agents)                         
    while True:
        actions = [policy1(states[0], False), policy2(states[1], False)]
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations         
        rewards = env_info.rewards                         
        dones = env_info.local_done                     
        scores += np.max(rewards)
        states = next_states                               
        if np.any(dones):                                  
            break
    
    return np.mean(scores)
    
# play a round to demonstrate agent
current_score = play_round(env, brain_name, agent1.act, agent1.act, config)    
print('score this round: {}'.format(current_score))

# train the agent and plot rewards
scores = ddpg(n_episodes=config.episode_count)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# play a round to demonstrate agent
current_score = play_round(env, brain_name, agent1.act, agent1.act, config)    
print('score this round: {}'.format(current_score))

env.close()
