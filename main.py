import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dqn import Agent, DQN


def run(episodes=2000, epsilon=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    '''
    Run until convergence
    '''
    env = gym.make('LunarLander-v2')
    scores, avg_scores = [], []
    np.random.seed(5)
    torch.manual_seed(5)
    env.seed(5)
    
    agent = Agent(gamma=0.99, batch_size=64, nactions=4, nstates=8, learning_rate=5e-4)
    for episode in range(1, episodes+1):
        state = env.reset()
        score = 0
        while True:
            action = agent.policy(state, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.add(state, action, reward, next_state, done)
            agent.optimize()
            state = next_state
            score += reward
            if done: break
        epsilon = max(epsilon_end, epsilon_decay*epsilon)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        print(f"\rEpisode {episode}, Average Score: {avg_score}", end="")
        if episode % 100 == 0:
            print(f"\rEpisode {episode}, Average Score: {avg_score}")
        if avg_score >= 200:
            print("Solved!")
            torch.save(agent.Q_local.state_dict(), 'model.pth')
            with open('scores_final.txt', 'w') as f:
                for item in scores:
                    f.write("%s\n" % item)
            with open('avg_scores_final.txt', 'w') as f:
                for item in avg_scores:
                    f.write("%s\n" % item)
                    
            ## plot and save fig
            plt.figure(figsize=(10, 6))
            plt.plot(scores)
            plt.plot(avg_scores, label='Average score of last 100 runs')
            plt.xlabel('Episodes', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.legend()
            plt.savefig('train.png', dpi=300)
            break
    return scores


def nodes_graphs(episodes=600):
    '''
    Hidden layer node count parameter testing
    '''
    nodes_avg_scores = {}
    nodes_scores = {}
    for nodes in [32, 64, 128, 256]:
        epsilon= 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        env = gym.make('LunarLander-v2')
        np.random.seed(5)
        torch.manual_seed(5)
        env.seed(5)
        scores, avg_scores = [], []
        agent = Agent(gamma=0.99, batch_size=64, nactions=4, nstates=8, learning_rate=5e-4, nodes=nodes)
        for episode in range(1, episodes+1):
            state = env.reset()
            score = 0
            while True:
                action = agent.policy(state, epsilon)
                next_state, reward, done, info = env.step(action)
                agent.add(state, action, reward, next_state, done)
                agent.optimize()
                state = next_state
                score += reward
                if done: break
            epsilon = max(epsilon_end, epsilon_decay*epsilon)
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
        nodes_avg_scores[nodes] = avg_scores
        nodes_scores[nodes] = scores
    
    ## plot and save fig
    plt.figure(figsize=(10, 6))
    for nodes, scores in nodes_avg_scores.items():
        plt.plot(scores, label=f"nodes = {nodes}")
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend()
    plt.savefig('nodes.png', dpi=300)


def gamma_graphs(episodes=600): 
    '''
    gamma parameter testing
    '''   
    gamma_avg_scores = {}
    gamma_scores = {}
    for gamma in [0.995, 0.99, 0.95, 0.9]:
        epsilon= 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        env = gym.make('LunarLander-v2')
        np.random.seed(5)
        torch.manual_seed(5)
        env.seed(5)
        scores, avg_scores = [], []
        agent = Agent(gamma=gamma, batch_size=64, nactions=4, nstates=8, learning_rate=5e-4)
        for episode in range(1, episodes+1):
            state = env.reset()
            score = 0
            while True:
                action = agent.policy(state, epsilon)
                next_state, reward, done, info = env.step(action)
                agent.add(state, action, reward, next_state, done)
                agent.optimize()
                state = next_state
                score += reward
                if done: break
            epsilon = max(epsilon_end, epsilon_decay*epsilon)
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
        gamma_avg_scores[gamma] = avg_scores
        gamma_scores[gamma] = scores
    
    ## plot and save fig
    plt.figure(figsize=(10, 6))
    for gamma, scores in gamma_avg_scores.items():
        plt.plot(scores, label=f"gamma = {gamma}")
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend()
    plt.savefig('gamma.png', dpi=300)


def decay_graphs(episodes=600):
    '''
    epsilon decay parameter testing
    '''
    decay_avg_scores = {}
    decay_scores = {}
    for epsilon_decay in [0.999, 0.995, 0.95, 0.9]:
        epsilon= 1.0
        epsilon_end = 0.01
        env = gym.make('LunarLander-v2')
        np.random.seed(5)
        torch.manual_seed(5)
        env.seed(5)
        scores, avg_scores = [], []
        agent = Agent(gamma=0.99, batch_size=64, nactions=4, nstates=8, learning_rate=5e-4)
        for episode in range(1, episodes+1):
            state = env.reset()
            score = 0
            while True:
                action = agent.policy(state, epsilon)
                next_state, reward, done, info = env.step(action)
                agent.add(state, action, reward, next_state, done)
                agent.optimize()
                state = next_state
                score += reward
                if done: break
            epsilon = max(epsilon_end, epsilon_decay*epsilon)
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
        decay_avg_scores[epsilon_decay] = avg_scores
        decay_scores[epsilon_decay] = scores
    
    ## plot and save fig
    plt.figure(figsize=(10, 6))
    for decay, scores in decay_avg_scores.items():
        plt.plot(scores, label=f"epsilon_decay = {decay}")
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend()
    plt.savefig('decay.png', dpi=300)


def learning_rate_graphs(episodes=600): 
    '''
    learning rate parameter testing
    '''   
    lr_avg_scores = {}
    lr_scores = {}
    for lr in [5e-3, 1e-3, 5e-4, 1e-4]:
        epsilon= 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        env = gym.make('LunarLander-v2')
        np.random.seed(5)
        torch.manual_seed(5)
        env.seed(5)
        scores, avg_scores = [], []
        agent = Agent(gamma=0.99, batch_size=64, nactions=4, nstates=8, learning_rate=lr)
        for episode in range(1, episodes+1):
            state = env.reset()
            score = 0
            while True:
                action = agent.policy(state, epsilon)
                next_state, reward, done, info = env.step(action)
                agent.add(state, action, reward, next_state, done)
                agent.optimize()
                state = next_state
                score += reward
                if done: break
            epsilon = max(epsilon_end, epsilon_decay*epsilon)
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
        lr_avg_scores[lr] = avg_scores
        lr_scores[lr] = scores
    
    ## plot and save fig
    plt.figure(figsize=(10, 6))
    for lr, scores in lr_avg_scores.items():
        plt.plot(scores, label=f"learning_rate = {lr}")
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend()
    plt.savefig('learning_rate.png', dpi=300)


def test():
    '''
    test run for 100 episodes
    '''
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, batch_size=64, nactions=4, nstates=8, learning_rate=5e-4)
    agent.Q_local.load_state_dict(torch.load('model.pth'))

    scores = []
    for x in range(100):
        state = env.reset()
        score = 0
        while True:
            action = agent.policy(state, epsilon=0)
            agent.Q_local.eval()
            next_state, reward, done, _ = env.step(action)
            agent.Q_local.eval()
            score += reward
            state = next_state
            if done:
                break
        scores.append(score)
        
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.savefig('test.png', dpi=300)

def main():
    print("Running until convergence...")
    run()
    print("Generating nodes graphs...")
    nodes_graphs(episodes=600)
    print("Generating gamma graphs...")
    gamma_graphs(episodes=600)
    print("Generating epsilon decay graphs...")
    decay_graphs(episodes=600)
    print("Generating learning rate graphs...")
    learning_rate_graphs(episodes=600)
    print("Generating test run...")
    test()
    print("Done!")

if __name__ == "__main__":
    main()