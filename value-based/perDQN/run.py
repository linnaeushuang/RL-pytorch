
import gym
import argparse
import os

from per_learner import learner
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_args():
    parser=argparse.ArgumentParser("Reinforcement learning experiments for DQN,discrete action space")
    # Environment
    parser.add_argument("--scenario", type=str, default="CartPole-v0", help="name of the scenario script")
    parser.add_argument("--train-model", type=int, default=1, help="train or test,train is 1,test is 0")
    parser.add_argument("--num-episodes", type=int, default=600, help="number of episodes")
    parser.add_argument("--model-path", type=str, default="nothing", help="path of the model to save or load")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.000625, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="number of episodes to optimize at the same time")
    parser.add_argument("--memory-size", type=int, default=4096, help="memory of the replaybuff")
    parser.add_argument("--target-update", type=int, default=32, help="update interval of target network")
    parser.add_argument("--epsilon-decay", type=float, default=0.001, help="decay of exploration rate")
    parser.add_argument("--start-epsilon", type=float, default=1.0, help="initial value of exploration rate")
    parser.add_argument("--end-epsilon", type=float, default=0.01, help="min value of exploration rate")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    return parser.parse_args()


def train(arglist):
    print("Train the agent")
    env=gym.make(arglist.scenario)
    np.random.seed(arglist.seed)
    env.seed(arglist.seed)
    torch.manual_seed(arglist.seed)

    agent=learner(env.observation_space.shape[0],
            env.action_space.n,
            arglist.memory_size,
            arglist.batch_size,
            arglist.target_update,
            arglist.lr,
            arglist.epsilon_decay,
            arglist.start_epsilon,
            arglist.end_epsilon,
            arglist.gamma,
            is_train=True)
    scores=[]
    episode=0
    while episode<=arglist.num_episodes:
        state=env.reset().astype(np.float32)
        done=False
        score=0
        while not done:
            action=agent.select_action(state)
            next_state, reward, done, info=env.step(action)
            next_state=next_state.astype(np.float32)
            agent.store_transition(state,action,reward,next_state,done)
            if agent.memory.size>arglist.batch_size:
                # this condition must exist,otherwise
                # the error :maximum recursion depth exceeded. will occur
                # the bug maybe in PrioritizedReplayBuffer.sample
                # line 62-65,  check the upperbound,and rewrite segment tree
                # fix this bug later
                # run.py can run
                agent.update_model()
            state=next_state
            score+=reward

        scores.append(score)
        episode+=1
    if arglist.model_path=='nothing':
        print("path error,model not save")
    else:
        agent.restore(arglist.model_path)
    env.close()
    return scores
            
def test(arglist):
    print("Test the agent")
    env=gym.make(arglist.scenario)
    np.random.seed(arglist.seed)
    env.seed(arglist.seed)
    torch.manual_seed(arglist.seed)
    agent=learner(env.observation_space.shape[0],
            env.action_space.n,
            arglist.memory_size,
            arglist.batch_size,
            arglist.target_update,
            arglist.lr,
            arglist.epsilon_decay,
            arglist.start_epsilon,
            arglist.end_epsilon,
            arglist.gamma,
            is_train=False)
    if os.path.exists(arglist.model_path):
        agent.load(arglist.model_path)
    else:
        print("path error")
        return None

    state=env.reset().astype(np.float32)
    done=False
    score=0

    while not done:
        action=agent.select_action(state)
        next_state,reward,done,_=env.step(action)
        state=next_state.astype(np.float32)
        score+=reward
    env.close()
    return score

def getAveScorePerEpisode(arglist):
    env=gym.make(arglist.scenario)
    train_agent=learner(env.observation_space.shape[0],
            env.action_space.n,
            arglist.memory_size,
            arglist.batch_size,
            arglist.target_update,
            arglist.lr,
            arglist.epsilon_decay,
            arglist.start_epsilon,
            arglist.end_epsilon,
            arglist.gamma,
            is_train=True)
    test_agent=learner(env.observation_space.shape[0],
            env.action_space.n,
            arglist.memory_size,
            arglist.batch_size,
            arglist.target_update,
            arglist.lr,
            arglist.epsilon_decay,
            arglist.start_epsilon,
            arglist.end_epsilon,
            arglist.gamma,
            is_train=False)
    with open('results.txt','wb') as results:
        scorelist=np.zeros(100)
        for epi in range(1,arglist.num_episodes+1):
            np.random.seed(arglist.seed)
            env.seed(arglist.seed)
            torch.manual_seed(arglist.seed)
            state=env.reset().astype(np.float32)
            done=False
            #-----------train-----------
            while not done:
                action=train_agent.select_action(state)
                next_state, reward, done, info=env.step(action)
                next_state=next_state.astype(np.float32)
                train_agent.store_transition(state,action,reward,next_state,done)
                if train_agent.memory.size>arglist.batch_size:
                # this condition must exist,otherwise
                    train_agent.update_model()
                state=next_state
            #-----------test-----------
            test_agent.net.load_state_dict(train_agent.net.state_dict())

            for test_epi in range(100):
                np.random.seed(test_epi+111)
                env.seed(test_epi+111)
                torch.manual_seed(test_epi+111)
                state=env.reset().astype(np.float32)
                done=False
                score=0
                while not done:
                    action=test_agent.select_action(state)
                    next_state, reward, done, info=env.step(action)
                    state=next_state.astype(np.float32)
                    score+=reward
                scorelist[test_epi]=score

            results.write(str(scorelist.mean())+'\t'+str(scorelist.std())+'\n')
            results.flush()



if __name__ == '__main__':
    # prevent cpu from being full
    torch.set_num_threads(2)
    arglist = parse_args()
    if arglist.train_model ==1:
        scores=train(arglist)
    elif arglist.train_model==2:
        scores=test(arglist)
    else:
        getAveScorePerEpisode(arglist)
