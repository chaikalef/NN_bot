# coding: utf-8

""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym

# hyperparameters
H = 30 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# model initialization
D = 2 * 1 # input dimensionality: 2 x 1 grid
if resume:
    model = pickle.load(open('pong_bot_kar_xy.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)
    
grad_buffer = { k : np.zeros_like(v) for k, v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k, v in model.iteritems() } # rmsprop memory


def sigmoid(x):
    # число -> число
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210 x 160 x 3 uint8 frame into one number """
    I = I[35:190, 20:140, 0] # crop 210x160x3 -> 155x120x1
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # ball just set to 1
    I = np.transpose(np.nonzero(I))
    
    if (I.size != 0):
        x, y = I[0, 0], I[0, 1]
        return [x / 155., y / 120.]
    else:
        return [0.5, 0.5]


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)  # or np.float64
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x) # матрица * вектор -> вектор
    h[h < 0] = 0 # ReLU nonlinearity
    logp = np.dot(model['W2'], h) # вектор * вектор -> число
    p = sigmoid(logp)
    return p, h # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel() # матрица * вектор -> вектор
    dh = np.outer(epdlogp, model['W2']) # вектор * вектор -> матрица
    dh[eph <= 0] = 0 # backpro prelu
    dW1 = np.dot(dh.T, epx) # матрица * матрица -> матрица
    return {'W1':dW1, 'W2':dW2}


env = gym.make("Pong-v0")
observation = env.reset()
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    if render:
        env.render()

    # preprocess the observation, set input to network to be difference image
    x = prepro(observation)

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    y = 1 if action == 2 else 0 # a "fake label"
    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

    if done: # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], [] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if (episode_number % batch_size == 0):
            for k, v in model.iteritems():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            pickle.dump(model, open('pong_bot_kar_xy.p', 'wb'))

        # boring book-keeping
        if (running_reward is None):
            running_reward = reward_sum
        else:
            running_reward = running_reward * 0.99 + reward_sum * 0.01
        
        #print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        
      #  reward_sum = 0
        observation = env.reset() # reset env

    #if .reward != 0.. . Pong has either +1 or -1 reward exactly when game ends.
        #print(('ep %d: game finished, re+ward: %f' % (episode_number, reward)) + 
            #  .'' if reward == -1 else ' !!!!!!!!'))
        open('pong_bot_kar_xy.csv', 'at').write(str(episode_number) + ',' + str(reward_sum) + ',' + str(running_reward) + '\n'); reward_sum = 0

