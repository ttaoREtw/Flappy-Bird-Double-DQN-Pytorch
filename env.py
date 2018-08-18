import gym
from gym import wrappers
import cv2

class FlappyBird(object):
    def __init__(self, width=84, height=84, outdir='tmp/result', seed=0, record_every_episode=None):
        try:
            import gym_ple
        except ImportError:
            raise ImportError('Maybe you could try "pip install gym_ple"')
        self.width = width
        self.height = height
        self.record_every_episode = record_every_episode
        self.env = gym.make('FlappyBird-v0')
        self.env.seed(seed)
        if record_every_episode:
            self.env = wrappers.Monitor(self.env, 
                directory=outdir, 
                force=True, 
                video_callable=lambda ep_id: ep_id % record_every_episode == 0)
        

    def get_screen(self, preprocess=True):
        screen = self.env.render(mode='rgb_array')
        if preprocess:
            screen = self.preprocess(screen)
        return screen

    def preprocess(self, screen):
        luma = [0.2989, 0.5870, 0.1140]
        revised = cv2.resize(screen, (self.height, self.width)) # resize
        revised = luma[0]*revised[:, :, 0] + \
                  luma[1]*revised[:, :, 1] + \
                  luma[2]*revised[:, :, 2]
        revised = revised.astype('float32') / 255.0
        return revised

    def step(self, action, preprocess=True):
        observation, reward, done, _ = self.env.step(action)
        if preprocess:
            observation = self.preprocess(observation)
        return observation, reward, done

    def reset(self):
        self.env.reset()
        self.env.step(action=1)

    def change_record_every_episode(self, record_every_episode):
        self.change_record_schedule(lambda ep_id: ep_id % record_every_episode == 0)

    def change_record_schedule(self, video_callable):
        self.env.video_callable = video_callable

    def close(self):
        self.env.close()

    @property
    def action_meaning(self):
        return ['flap', 'none']
    

