import gym
from reinforce import Reinforce
import matplotlib.pyplot as plt
import numpy as np
import os





def main():
    log_path = './reinforce_data/'
    test_interval = 5
    RENDER = False
    test_loss_mean = []
    test_loss_min = []
    test_loss_max = []

    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    RL = Reinforce(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.02,
        reward_decay=0.99,
        # output_graph=True,
    )

    for i_episode in range(3000):
        observation = env.reset()
        while True:
            if RENDER: env.render()
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            #print(action,reward)
            RL.store_transition(observation, action, reward)
            if done:
                ep_rs_sum = sum(RL.ep_rs)
                #if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", ep_rs_sum)
                vt = RL.learn()
                if i_episode % test_interval ==0:
                    test_loss = []
                    for _ in range(100):
                        reward_total = 0
                        observation = env.reset()
                        while True:
                            action = RL.choose_action(observation)
                            observation_, reward, done, info = env.step(action)
                            reward_total = reward_total + reward
                            observation = observation_
                            if done:
                                test_loss.append(reward_total)
                                break
                    test_loss = np.array(test_loss,dtype = np.float32)
                    test_loss_mean.append(np.mean(test_loss))
                    test_loss_min.append(np.min(test_loss))
                    test_loss_max.append(np.max(test_loss))
                    np.save(log_path+'mean.npy',test_loss_mean)
                    np.save(log_path + 'min.npy', test_loss_min)
                    np.save(log_path + 'max.npy', test_loss_max)

                    print(np.mean(test_loss),np.min(test_loss),np.max(test_loss))
                break

            observation = observation_



if __name__=="__main__":
    main()