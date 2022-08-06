import gym
import numpy as np
from DQN import DQNAgent


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    load_checkpoint = False
    best_score = -np.inf
    n_games = 10000
    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.001,
                     state_dims=env.observation_space.shape[0], num_actions=env.action_space.n, action_dims=1)

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0.0
        while not done:
            action = agent.take_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.memorize(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        if avg_score > best_score:
            best_score = avg_score
        print('episode: ', i, 'score: ', score,
              'average score %.1f' % avg_score, 'best score %.2f' % best_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

