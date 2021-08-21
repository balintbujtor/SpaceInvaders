"""
Autonomous and Adaptive Systems Project 2021
Comparing Deep Q Learning and Double Deep Q Learning in Space Invaders

Balint Bujtor
Universita di Bologna
balint.bujtor@studio.unibo.it

During the development of the code I used the following sites/tutorials for inspiration and help:

https://github.com/VforV93/Q-Learning-vs-Deep-Q-Learning-in-continuous-state-space
https://github.com/gandroz/rl-taxi/blob/main/pytorch/agent.py
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://towardsdatascience.com/optimized-deep-q-learning-for-automated-atari-space-invaders-an-implementation-in-tensorflow-2-0-80352c744fdc
https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

"""

import gym
import numpy
import random
import torch
import cv2
import os
import datetime
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plot
from collections import deque

# initializing the pseudo random generators to use the same values
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Stack(object):
    """
    this class serves as a container for the stacked frames
    contains num_of_frames states at the same time
    also executes the image preprocessing before returning the stack
    """
    def __init__(self, frames=4, w=80, h=105):
        self.num_of_frames = frames
        self.w = w
        self.h = h
        self.stacked_state = None
        self.stacked_frames = deque([numpy.zeros((self.h, self.w), dtype=numpy.int) for _ in range(self.num_of_frames)],
                                    maxlen=self.num_of_frames)

    def do_stacking(self, cur_frame, new_ep):

        # preprocessing current image
        cur_frame = image_preprocessing(cur_frame)

        if new_ep:
            # clearing deque if it is a new ep
            self.stacked_frames = deque(
                [numpy.zeros((self.h, self.w), dtype=numpy.int) for _ in range(self.num_of_frames)],
                maxlen=self.num_of_frames)

            # maxing the two most recent frames as in the paper
            max_frame = numpy.maximum(cur_frame, cur_frame)

            for _ in range(self.num_of_frames):
                self.stacked_frames.append(max_frame)

            self.stacked_state = numpy.stack(self.stacked_frames, axis=2)

        else:
            maxframe = numpy.maximum(self.stacked_frames[-1], cur_frame)

            self.stacked_frames.append(maxframe)

            self.stacked_state = numpy.stack(self.stacked_frames, axis=2)

        # stacked state is a numpy array of (4, 105, 80) the stack is a deque of the last 4 elements
        return self.stacked_state, self.stacked_frames


class ReplayBuffer(object):
    """
    this class serves as a container for the step information
    stacked frames (states), action, reward, next_state (one), done

    adding, sampling of the episodes are implemented
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def add_one(self, s, a, r, ns, d):
        # add a stack of states/screens, the action, reward, next_state, done
        self.memory.append(numpy.array([s, a, r, ns, d]))

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)

        samples = numpy.array(samples)

        states, actions, rewards, next_states, finished = [], [], [], [], []

        for experience in samples:
            states.append(experience[0])
            actions.append(experience[1])
            rewards.append(experience[2])
            next_states.append(experience[3])
            finished.append(experience[4])

        # returning samples by their category
        return states, actions, rewards, next_states, finished

    def __len__(self):
        return len(self.memory)


class Qnet(nn.Module):
    """
    the deep Q network
    """
    def __init__(self, convolution, action_shape, kernel_size=3, w=80, h=105):
        super(Qnet, self).__init__()
        self.conv_size = convolution
        self.num_actions = action_shape
        self.kernel_size = kernel_size
        self.w = w
        self.h = h
        self.conv1 = nn.Conv2d(4, convolution, kernel_size)
        self.conv2 = nn.Conv2d(convolution, convolution, kernel_size)
        self.conv3 = nn.Conv2d(convolution, convolution, kernel_size)

        def conv2d_size_out(size, kernel=kernel_size, stride=1):
            return (size - (kernel - 1) - 1) // stride + 1

        # three times because we have three layers
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        fc_input_size = convw * convh * convolution

        self.fc = nn.Linear(fc_input_size, action_shape)

    def forward(self, x):
        x = x.transpose((0, 3, 1, 2))  # transposing so that number of stacked frames is the first dimension
        x = torch.from_numpy(x)  # transforming it into tensor
        x = x.float()
        x = x.to(device)

        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def plot_save_episode_measures(my_array, name):
    """
    function structure from this tutorial:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    :param my_array: the array of metrics to evaluate
    :param name: name of the figure
    :return: saved plots in the util folder
    """

    global folder

    plot.figure(2)
    plot.clf()
    measure = torch.tensor(my_array, dtype=torch.float)
    plot.xlabel('Episode')
    plot.ylabel(name)
    plot.plot(measure.numpy())
    datee = datetime.datetime.now().strftime("%m_%d_%H_%M")
    plot.savefig(os.path.join(folder, '{0}_{1}.png'.format(name, datee)))

    # take 100 episode avgs and plot them
    if len(measure) >= 100:
        means = measure.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plot.plot(means.numpy())
        plot.savefig(os.path.join(folder, 'means_{0}_{1}.png'.format(name, datee)))

        print(means.numpy())

    plot.pause(0.001)


def image_preprocessing(img):
    """
    image resize (halfing), converting to grayscale
    :param img: input image (one image)
    :return: preprocessed image
    """

    img = cv2.resize(img, dsize=(80, 105))

    img = numpy.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    # (105, 80) grayscale

    # no cropping because at the place of the score sometimes an invader
    # appears which shoots as well

    # plot.imshow(img, cmap='gray')
    # plot.show()

    return img


def epsilon_greedy_policy(cur_state, net, stepp):
    """
    choosing an action based on epsilon greedy policy
    :param cur_state: current state (stack of inputs)
    :param net: dqn network which computes the action
    :param stepp: which step we are to calculate the epsilon
    :return: the chosen action
    """
    epsilon = EPS_MIN + (EPS_MAX - EPS_MIN) * numpy.exp(-1.0 * stepp / EPS_DECAY)

    if random.random() < epsilon:
        rand_act = random.randint(0, net.num_actions - 1)
        return rand_act
    else:
        q_values = net(cur_state[numpy.newaxis])
        q_values = q_values.max(1)[1]
        return q_values


def play_one_step(net, buffer, losss, optimizer, is_double, target_net, batch_size, discount_rate):
    """
    playing one training step of the net
    choosing BATCH_SIZE examples from the buffer
    predicting next action with the net without using gradient
    calculating target qs and then calculating loss and optimizing the net

    :param net: deep q net
    :param buffer: replay buffer
    :param losss: loss
    :param optimizer: optimizer for loss
    :param is_double: if we should calculate target q-s with tartget network
    :param target_net: the target network for target q-s
    :param batch_size: batch size
    :param discount_rate discount rate for policy
    """

    states, actions, rewards, next_states, finishes = buffer.sample(batch_size)

    # no target network because target q value is calculated without another network
    # calculating the next actions from next states to determine target Q values

    target_qs = []
    if is_double:
        next_actions = target_net(numpy.array(next_states)).max(1)[1]

        for item in range(batch_size):
            target_qs.append((1 - finishes[item])*next_actions[item]*discount_rate + rewards[item])
    else:
        with torch.no_grad():
            # next qs are needed to calculate the target q, but without back propagation
            # max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            next_actions = net(numpy.array(next_states)).max(1)[1]

        # getting the target Q-s for each item in the batch
        for item in range(batch_size):
            target_qs.append(rewards[item] + (1 - finishes[item])*discount_rate*next_actions[item])

    mask = f.one_hot(torch.tensor(actions, device=device), net.num_actions)

    # actual q is needed to calculate loss we need the grad to perform optimization
    all_qs = net(numpy.asarray(states, dtype=numpy.float))

    predicted_qs = torch.sum(all_qs*mask, dim=1, keepdim=True).to(device)

    actual_loss = losss(predicted_qs, torch.tensor(target_qs, device=device).unsqueeze(1))

    optimizer.zero_grad()
    actual_loss.backward()

    # gradient clipping to avoid gradient explosion
    nn.utils.clip_grad_value_(net.parameters(), 1.0)

    optimizer.step()

    return actual_loss


STACK_SIZE = 4
TRAINING = 800
WARMUP = 50
STEP_SIZE = 1200
EPS_MAX = 0.9
EPS_MIN = 0.05
EPS_DECAY = 300

'''
TARGET_UPDATE = 10
REPLAY_CAPACITY = 20000
BATCH_SIZE = 16
CONV_SIZE = 16
KERNEL_SIZE = 5
DISCOUNT_RATE = 0.95
LEARNING_RATE = 0.001
'''

main_config = {
                "is_double": True,
                "TARGET_UPDATE": 20,            # 10, 20, 30
                "REPLAY_CAPACITY": 40000,       # 40000
                "BATCH_SIZE": 8,                # 8
                "CONV_SIZE": 16,                # 16
                "KERNEL_SIZE": 3,               # 3
                "DISCOUNT_RATE": 0.99,          # 0.99
                "LEARNING_RATE": 1e-5,          # 1e-5
}

cwd = os.getcwd()
folder = os.path.join(cwd, 'files')


def train(config):
    """
    executes the training of the deep q network
    filling up the replay buffer then executing an optimization after each step in the game
    also plotting and saving measures and models
    :param: config: general config
    :return: saved best model, saved plots and files of the measures
    """
    episode_durations = []
    episode_rewards = []
    best_reward = 0

    env = gym.make('SpaceInvaders-v0')
    env.seed(42)
    # v4: always does my action while v0 has 25% chance of doing the previous action
    # deterministic: always skips 4 frames, without deterministic, it samples from (2, 5)
    # I cannot use deterministic for this game, since the frequency of the bullets
    # are eliminated with 4 frame skips

    # STATE_SHAPE = env.observation_space.shape
    action_shape = env.action_space.n

    stack = Stack(STACK_SIZE)
    replay_buffer = ReplayBuffer(config["REPLAY_CAPACITY"])
    q_network = Qnet(config["CONV_SIZE"], action_shape, kernel_size=config["KERNEL_SIZE"]).to(device)

    try:
        q_network.load_state_dict(torch.load(os.path.join(folder, 'best_model.pt')))
        print("loaded best saved model")
    except FileNotFoundError:
        print('no best saved model')

    target_q_network = None
    if config["is_double"]:
        target_q_network = Qnet(config["CONV_SIZE"], action_shape, kernel_size=config["KERNEL_SIZE"]).to(device)
        target_q_network.load_state_dict(q_network.state_dict())
        target_q_network.eval()

    loss = nn.SmoothL1Loss()
    optimizer = optim.Adam(q_network.parameters(), lr=config["LEARNING_RATE"])

    for game in range(WARMUP + TRAINING):
        state = env.reset()

        obs, _ = stack.do_stacking(state, True)

        game_reward = 0
        game_loss = 0
        plot_step = 1
        for step in range(STEP_SIZE):

            # playing randomly in the beginning
            action = epsilon_greedy_policy(obs, q_network, step)
            next_obs, reward, done, _ = env.step(action)

            next_obs, stacked_frames = stack.do_stacking(next_obs, False)

            replay_buffer.add_one(obs, action, reward, next_obs, done)

            obs = next_obs
            # env.render()

            game_reward += reward
            if game == 0:
                best_reward = game_reward

            if done:
                # if we finish a game then we save the episode durations and the plots and quit
                episode_durations.append(plot_step)
                episode_rewards.append(game_reward)
                break

            if game > WARMUP:
                # if we played 50 games (filled replay buffer) we start to train the network on the replay buffer
                act_loss = play_one_step(q_network, replay_buffer, loss, optimizer, config["is_double"],
                                         target_q_network, config["BATCH_SIZE"], config["DISCOUNT_RATE"])
                game_loss += act_loss

            plot_step += 1

        if config["is_double"] and game % config["TARGET_UPDATE"] == 0:
            target_q_network.load_state_dict(q_network.state_dict())

        if game > WARMUP and game_reward > best_reward:
            # we save the weights every fifty steps or when we have better game reward
            if not os.path.exists(folder):
                os.makedirs(folder)
            torch.save(q_network.state_dict(), os.path.join(folder, 'weights_%d.pt' % game))
            best_reward = game_reward

            if config["is_double"] is True:
                torch.save(target_q_network.state_dict(), os.path.join(folder, 'target_weights_%d.pt' % game))

        date = datetime.datetime.now().strftime("%m_%d")

        # we also save the episode lengths to a text file
        with open(os.path.join(folder, 'episode_lengths_%s.txt' % date), 'a+') as text_file:
            text_file.write(str(plot_step) + '\n')

        with open(os.path.join(folder, 'episode_rewards_%s.txt' % date), 'a+') as text_file:
            text_file.write(str(game_reward) + '\n')

    plot_save_episode_measures(episode_durations, 'durations')
    plot_save_episode_measures(episode_rewards, 'rewards')

    env.close()


def test(config):
    """
    demo with the trained model

    :return: nothing
    """

    env = gym.make('SpaceInvaders-v0')
    env.seed(42)

    action_shape = env.action_space.n

    stack = Stack(STACK_SIZE)
    q_network = Qnet(config["CONV_SIZE"], action_shape, kernel_size=config["KERNEL_SIZE"]).to(device)
    q_network.eval()

    try:
        q_network.load_state_dict(torch.load(os.path.join(folder, 'best_model.pt')))
        print("loaded best saved model")
    except FileNotFoundError:
        print('no best saved model')

    state = env.reset()

    obs, _ = stack.do_stacking(state, True)
    done = False

    while not done:
        # playing randomly in the beginning
        action = q_network(obs[numpy.newaxis]).max(1)[1]
        next_obs, reward, done, _ = env.step(action)

        next_obs, stacked_frames = stack.do_stacking(next_obs, False)

        obs = next_obs
        env.render()

    env.close()


def main():
    """
    main function implementing training of testing
    :return: trained or tested model and optionally the files
    """

    do_train = False
    do_test = True

    if do_train is True and do_test is False:

        if main_config["is_double"] is True:
            print("Training with double dqn")
        else:
            print("Training with dqn")

        train(main_config)

    elif do_test is True and do_train is False:
        test(main_config)

    else:
        print("wrong param init")


if __name__ == "__main__":
    main()
