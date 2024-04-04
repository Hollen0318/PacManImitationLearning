import gym
import os
import time
import torch
import random
import utils
env_seed = random.randint(0, 9999)
import datetime

def run_recorder(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # We create the environment so it's new and read the observations saved into obs
    env = gym.make("ALE/MsPacman-v5", render_mode='human')
    obs = env.reset(seed=env_seed)[0]
    print("Welcome to the expert recorder")
    print("To record press:")
    print("w: turn up")
    print("a: turn left")
    print("s: turn down")
    print("d: turn right")
    print("Once you're finished press p to save the data")
    print("NOTE: Make sure you've selected the console window in order for the application to receive your input.")
    total_reward = 0.0
    action = 3
    esc = False
    start_moving = False
    record_freq = 2
    while not start_moving:
        action = 3
        next_obs, reward, done, another_done, info = env.step(action)
        if reward != 0.0:
            total_reward += reward
            start_moving = True
            time_stamp = 0
            learn_pairs = {}
            obs = utils.preprocess_observation(next_obs)
            learn_pairs[time_stamp] = (obs, torch.tensor(action))
            print("time stamp = ", time_stamp, " action = ", action, " reward = ", reward, " done = ", done, " another_done = ", another_done, " info = ", info)

    while not esc:
        done = False
        while not done:
            time_stamp += 1
            last_action = action
            action = utils.key_to_action()
            if action == 0:
                action = last_action
            if action == 999:
                esc = True
                break
            next_obs, reward, done, another_done, info = env.step(action)
            total_reward += reward
            act = torch.tensor(int(action))
            if time_stamp%record_freq == 0:
                print("time stamp = ", time_stamp, " action = ", action, " reward = ", reward, " done = ", done, " another_done = ", another_done, " info = ", info)
                learn_pairs[time_stamp//record_freq] = (obs, act)
            obs = utils.preprocess_observation(next_obs)
            time.sleep(0.06)
        print("you died!")
        esc = True
    env.close()
    print("SAVING")
    # print(learn_pairs)
    filename = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S.pt")
    save_path = save_path + "/Expert_" + filename
    torch.save(learn_pairs, save_path)
    print("You saved the file into location = ", save_path)
    os.chmod(save_path, 0o600)
    # Load the saved dictionary
    try:
        loaded_data_dict = torch.load(save_path)
        print("File loaded successfully")
        # print(loaded_data_dict)
    except Exception as e:
        print("Error loading the file:")
        print(e)
    reward_record = str(total_reward) + " " + filename
    utils.log_reward(int(total_reward))
    print("Your total reward in this run is = ", total_reward)
save_path = "saved_episode"
run_recorder(save_path)