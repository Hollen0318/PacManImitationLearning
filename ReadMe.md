# Pacman-v5 Project
This project utilizes the imitation learning to train an agent to play the Pacman-v5 game, with the average awards per episode around 2000. 

## Dependencies
Please install all the packages listed in the dependencies.yml to make sure smooth reproduction. 

## Training
python train.py
The data will be loaded under the saved_episode.

## Evaluation instructed by Professor Sam Wiseman
python eval.py 
or visualize_eval.ipynb
(Make sure you modify the model_path inside the eval.py if you want to evaluate your own model)

## Generate training datasets
python record.py
The data will be stored under the saved_episode.

## Visualization Evaluation 
python visualize_play.py

## File description
model.py provides the model ImitationCNNNet and PolicyNetwork
