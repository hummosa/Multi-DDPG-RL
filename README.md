# Policy-Gradient-Classics-pytorch

This repository is an implementation of the reinforcement learning algorithm DDPG as described by [Lillicrap et. al](https://arxiv.org/abs/1509.02971), with multiple instantiations of the DDPG agent to tackle basic multi-agent RL problems. 

#### Unity [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment:

This the third project of the Udacity deep RL nanodegree and it aims to solve a version of the Unity Tennis environment, by achieving a score of >0.5 over 100 consecutive episodes, taking the max score achieved by either of the two agents.

The environment models a simple tennis table with two controllable rackets on either side. An agent is rewarded +0.1 if it hits the ball over the net, and is rewarded -0.01 if it lets the ball go out of bounds or drops it. We approach this environment as a multi-agent setup with one agent for each Tennis racket. Each agent receives as an observation an 8 dimensional vector (with the 3 most recent frames stacked). Each agent outputs 2 continuous action values corresponding to moving the racket towards the net, or jumping up.


The environment is episodic with a continuous state space and a continuous action space. It’s a generally stationary environment, expect for the policy of the other agent. The action space is continuous, which constrain our choice of algorithm to solve it.  



To run the enviornment locally a few required packages are necessary.

1) Udacity provides [this github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) with many of the dependencies required to run the environment. Follow the instructions in the readme.md file under thea heading 'dependencies' to clone a copy and install the required packages. 
     * Note also that it will be required to install ipykernel (pip install ipykernel) to execute all the steps.

2) Installing Unity enviroments is an option, but it will be easier to download only the required executable provided by Udacity. [The Windows 64bit compabtible version](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip) is included here for reference, but other versions can be found on the Udacity github repository.

3) To run the enviroment locally additional Unity packages are required as detailed [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md). 
    * **Note:** it is also important to run 'pip install unityagents' which is not mentioned in the instructions.
    
Training and demo:
The file tennis_2ddpg.py is the main python file used to load the environment, instantiate and train and agent. The function play_round can be used to demonstrate a learned agent.

Weight files are also included for a fully trained agent.
Here are the average scores over training episodes:

![Training curve](https://raw.githubusercontent.com/hummosa/Policy-Gradient-Classics-pytorch/master/DDPG_training_scores.png)

