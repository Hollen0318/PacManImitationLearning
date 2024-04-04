import argparse
from model import ImitationCNNNet
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os
import datetime

import utils

# The network architecture

o_s = []
a_s = []
for exp_f in os.listdir('saved_episode'):
    if not exp_f.startswith('Expert') or not exp_f.endswith('.pt'):
        continue
    path = 'saved_episode/' + exp_f
    print("Reading ", exp_f)
    os.chmod(path, 0o600)
    o_a = list(torch.load(path).values())
    o = torch.stack([item[0] for item in o_a])
    a = torch.stack([item[1] for item in o_a])
    # print("shape of o = ", o.shape)
    # print("shape of a = ", a.shape)
    o_s.append(o)
    a_s.append(a)
observations = torch.cat(o_s, dim=0)
actions = torch.cat(a_s, dim=0)
print("observations shape = ", observations.shape)
print("actions shape = ", actions.shape)

n_actions = 9
h, w = 210, 160  # height and width of the input images
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

lowest_loss = float('inf')  # Set initial lowest loss to infinity
# Instantiate the model and optimizer
model = ImitationCNNNet(n_actions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
from torch.utils.data import DataLoader, TensorDataset
train_dataset = TensorDataset(observations, actions)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    prev_state = None
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, prev_state = model(inputs, prev_state)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    epoch_loss = running_loss / (i + 1)
    # if epoch % 10 == 0:
        # print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    if epoch_loss < lowest_loss:
        lowest_loss = epoch_loss
        model_path = "saved_model"
        filename = datetime.datetime.now().strftime("best_model_%Y-%m-%d.pth")
        os.makedirs(model_path, exist_ok=True)  # Create the directory if it doesn't exist
        torch.save(model.state_dict(), os.path.join(model_path, filename))
        print(f'Saving model at epoch {epoch+1} with loss {epoch_loss:.4f} at {filename}')
    
print("Finished Training")

import gymnasium as gym
import os
import random
env_seed = random.randint(0, 9999)
import datetime

def run_evaluation(model):
    # We create the environment so it's new and read the observations saved into obs
    env = gym.make("ALE/MsPacman-v5")
    env_seed = random.randint(0, 9999)
    obs = env.reset(seed=env_seed)[0]
    obs = utils.preprocess_observation(obs).unsqueeze(0).to(device)
    # print("obs = ", obs.shape)
    total_reward = 0.0
    learn_pairs = {}
    time_stamp = 0
    done = False
    prev_state = None
    while not done:
        action, prev_state = model.get_action(obs, prev_state)
        next_obs, reward, done, another_done, info = env.step(action)
        total_reward += reward
        # print("action = ", action, " reward = ", reward, " done = ", done, " another_done = ", another_done, " info = ", info)
        learn_pairs[time_stamp] = (action, reward, total_reward)
        time_stamp += 1
        obs = utils.preprocess_observation(next_obs).unsqueeze(0).to(device)
    # print("you died!")
    env.close()
    return total_reward

parser = argparse.ArgumentParser()

parser.add_argument("--eval", type=bool, default=True, help="whether to evaluate the trained model")

args = parser.parse_args()
if args.eval:
    eval_model = ImitationCNNNet(n_actions).to(device)
    eval_model.load_state_dict(torch.load('saved_model/best_model_2023-04-30.pth'))
    eval_model.eval()
    save_path = "saved_model"
    running_avg = 0.0
    epr_num = 200
    for idx in range(epr_num+1):
        new_reward = run_evaluation(eval_model)
        running_avg += (new_reward - running_avg) / (idx+1)
        running_avg = round(running_avg, 2)
        if idx % 20 == 0:
            print(f"Until Evaluation # {idx} Average reward = ", running_avg)