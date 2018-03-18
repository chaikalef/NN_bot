# coding: utf-8

""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym

# hyperparameters
H = 30 # number of hidden layer neurons
learning_rate = 1e-4
resume = False # resume from previous checkpoint?
render = False
episode_number = 0

# model initialization
if resume:
    model = pickle.load(open('pong_bot_cha.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H) # "Xavier" initialization
    model['W2'] = np.random.randn(H, H) / np.sqrt(H)
    model['W3'] = np.random.randn(H) / np.sqrt(H)


def sigmoid(x, deriv = False):
    f = 1.0 / (1.0 + np.exp(-x))
    if (deriv == False):
        return f
    else:
        return f * (1 - f)


def prepro(I):
    """ prepro 210 x 160 x 3 uint8 frame into one number """
    I = I[35:190, 20:140, 0] # crop 210x160x3 -> 155x120x1
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = np.transpose(np.nonzero(I))
    
    if (I.size != 0):
        x = I[0, 0]
        return x / 155.
    else:
        return 0.5


def policy_forward(x):
    l1 = x * model['W1'] # число * вектор -> вектор
    l1[l1 <= 0] = 0 # ReLU nonlinearity
    l2 = np.dot(model['W2'], l1) # матрица * вектор -> вектор
    l2[l2 <= 0] = 0 # ReLU nonlinearity
    l3 = np.dot(model['W3'], l2) # вектор * вектор -> число
    l3 = sigmoid(l3)
    return l1, l2, l3 # return probability of taking action 2, and hidden state


env = gym.make("Pong-v0")
observation = env.reset()
coords, l1s, l2s, loss_cache1, loss_cache2 = [], [], [], [], []

while True:
    
    if render:
        env.render()

    # preprocess the observation, set input to network to be ball coordinates
    coord = prepro(observation)

    # forward the policy network and sample an action from the returned probability
    l1, l2, l3 = policy_forward(coord)
        
    if (l3 > 0.5):
        action = 2
    elif (l3 < 0.5):
        action = 3
    else:
        action = 1

    # record various intermediates (needed later for backprop)
    coords.append(coord) # observation
    l1s.append(l1) # hidden state
    l2s.append(l2) # hidden state

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    
    if (reward != 0): # an episode finished
        episode_number += 1

        # loss for output layer
        output_loss = reward
        
        # robocraft.ru/blog/algorithm/560.html
        # loss for hidden layer
        loss_cache2 = output_loss * model['W3']
        loss_cache1 = np.dot(model['W2'], loss_cache2)
        
        # edit each neuron in layer in loop
        # new_weiht[i] = old_weight[i] +
        #          + learning_rate * 
        #                  * loss_for_this_layer[i] * 
        #                  * y[i] *
        #                  * d(f(e[i])) / d(e[i])
        # i - index of each neuron in layer
        # y[i] - output signal of previous neuron on this edge
        # e[i] - input signal (взвешенная sum) of this neuron
        # f(e[i]) - activition function of input signal
        # d(f(e[i])) / d(e[i]) - deviration of f(e[i])
        for i in range(H):
            # edit weihts between input layer and hidden layer
            # f = ReLU
            # d(f(e[i])) / d(e[i]) = 0 or 1
            for j in range(len(coords)):
                if (model['W1'][i] > 0):
                    model['W1'][i] += learning_rate * loss_cache1[i] * coords[j]
                
            # edit weihts between first hidden layer and second hidden layer
            # f = ReLU
            # d(f(e[i])) / d(e[i]) = 0 or 1
            for j in range(len(l1s)):
                for edge in range(H):
                    if (model['W2'][i][edge] > 0):
                     model['W2'][i][edge] += learning_rate * loss_cache2[i] * l1s[j][edge]
                
            # edit weihts between second hidden layer and output layer
            # f = sigmoid
            # d(f(e[i])) / d(e[i]) = f * (1 - f)
            for j in range(len(l2s)):
                model['W3'][i] += learning_rate * output_loss * sigmoid(np.dot(model['W3'], l2s[j]), deriv = True) * l2s[j][i]

        pickle.dump(model, open('pong_bot_cha.p', 'wb'))

        observation = env.reset() # reset env
        coords, l1s, l2s, loss_cache1, loss_cache2 = [], [], [], [], [] # reset array memory
        
        # Pong has either +1 or -1 reward exactly when game ends.
        #print(('Ep %d, game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
        open('pong_bot_cha.csv', 'at').write(str(episode_number) + ',' + str(reward) + '\n')

