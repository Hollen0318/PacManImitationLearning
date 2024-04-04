import gym
import time
import torch
import random
from model import ImitationCNNNet
env_seed = random.randint(0, 9999)
import torch
from torch import nn
from torch.nn import functional as F
from utils import preprocess_observation
from torch.distributions import Categorical

def run_visualizer(model):
    # We create the environment so it's new and read the observations saved into obs
    env = gym.make("ALE/MsPacman-v5", render_mode='human')
    obs = env.reset(seed=env_seed)[0]
    obs = preprocess_observation(obs).unsqueeze(0).to(device)
    total_reward = 0.0
    esc = False
    done = False
    prev_state = None
    while not done:
        action, prev_state = model.get_action(obs, prev_state)
        next_obs, reward, done, another_done, info = env.step(action)
        total_reward += reward
        print("action = ", action, " reward = ", reward, " done = ", done, " another_done = ", another_done, " info = ", info)
        obs = preprocess_observation(next_obs).unsqueeze(0).to(device)
        time.sleep(0.06)
    print("you died!")
    env.close()
    print("Your total reward in this run is = ", total_reward)

n_actions = 9
device = "cpu"

eval_model = ImitationCNNNet(n_actions).to(device)
eval_model.load_state_dict(torch.load('saved_model/best_model_2023-04-30.pth', map_location=torch.device('cpu')))
eval_model.eval()
run_visualizer(eval_model)