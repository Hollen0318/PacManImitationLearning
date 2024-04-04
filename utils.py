import logging
import torch
import numpy as np
import gymnasium as gym
import keyboard
import os
import re

logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)

try:
    import matplotlib.pyplot as plt
    can_render = True
except:
    logging.warning("Cannot import matplotlib; will not attempt to render")
    can_render = False


key_map = {
    'w': 1,     # Up
    's': 4,     # Down
    'a': 3,     # Left
    'd': 2,     # Right
    'p': 999,   # Esc
}

def log_reward(reward, filename="saved_episode/reward_log.txt"):
    # If file doesn't exist, start with game 1
    if not os.path.exists(filename):
        game_no = 1
    else:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # If file is empty, start with game 1
        if not lines:
            game_no = 1
        else:
            # Extract last game number and increment it
            last_line = lines[-1]
            match = re.search(r'game(\d+):', last_line)
            if match:
                game_no = int(match.group(1)) + 1
            else:
                print("Unexpected file format. Please check the file.")
                return

    # Append the new game reward to the file
    with open(filename, 'a') as f:
        f.write(f"game{game_no}: {reward}\n")

def key_to_action():
    for key, action in key_map.items():
        if keyboard.is_pressed(key):
            return action
    return 0  # No action


def preprocess_observation(obs):
    """
    obs - a 210 x 160 x 3 ndarray representing an atari frame
    returns:
      a 3 x 210 x 160 normalized pytorch tensor
    """
    return torch.from_numpy(obs).permute(2, 0, 1)/255.0


def validate(model, render=False, nepisodes=1):
    print("render = ", render)
    print("nepisodes = ", nepisodes)
    assert hasattr(model, "get_action")
    torch.manual_seed(590060)
    np.random.seed(590060)
    model.eval()

    render = render and can_render

    if render:
        nepisodes = 1
        fig, ax = plt.subplots(1, 1)

    total_reward = 0
    steps_alive = []
    for i in range(nepisodes):
        env = gym.make("ALE/MsPacman-v5")
        obs = env.reset(seed=590060+i)[0]
        if render:
            im = ax.imshow(obs)
        observation = preprocess_observation( # 1 x 1 x ic x iH x iW
            obs).unsqueeze(0)
        prev_state = None
        step = 0
        # play until the agent dies or we exceed 50000 observations
        while env.ale.lives() == 3 and step < 50000:
            action, prev_state = model.get_action(observation, prev_state)
            # print("action = ", action)
            env_output = env.step(action)
            total_reward += env_output[1]
            if render:
                img = env_output[0]
                im.set_data(img)
                fig.canvas.draw_idle()
                plt.pause(0.1)
            observation = preprocess_observation(
                env_output[0]).unsqueeze(0)
            step += 1
        steps_alive.append(step)

    logging.info("Steps taken over each of {:d} episodes: {}".format(
        nepisodes, ", ".join(str(step) for step in steps_alive)))
    logging.info("Total return after {:d} episodes: {:.3f}".format(nepisodes, total_reward))