from utils import validate
from model import ImitationCNNNet
import torch
import argparse
n_actions = 9
device = "cpu"
parser = argparse.ArgumentParser()

parser.add_argument("--render", default=False, help="render game-play at validation time")
parser.add_argument("--nepisodes", default=2, help="number of episodes")

args = parser.parse_args()
eval_model = ImitationCNNNet(n_actions).to(device)
eval_model.load_state_dict(torch.load('saved_model/best_model_2023-04-30.pth'))
validate(eval_model, args.render, args.nepisodes)