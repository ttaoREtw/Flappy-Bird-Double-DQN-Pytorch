# Flappy-Bird-Double-DQN-Pytorch
Train an agent to play flappy bird game using double DQN model and implement it with pytorch.

![Result](result.gif)


## Installations (suggest using virtualenv)
* Pygame
```bash
$ pip install pygame
```
* ple
```bash
$ git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
$ cd PyGame-Learning-Environment/
$ pip install -e .
```
* gym
```bash
$ pip install gym
```
* gym-ple
```bash
$ pip install gym-ple
```

## How to run
* Train
```bash
# If cuda is available, add --cuda Y.
# Add --ckpt [ckpt_file] to train from this checkpoint. 
$ python main.py --mode train 
```
* Evaluation
```bash
# If cuda is available, add --cuda Y.
$ python main.py --mode eval --ckpt [checkpoint.pth.tar]
```
