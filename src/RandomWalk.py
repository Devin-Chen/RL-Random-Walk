import random
import numpy as np
class RandomWalk():
    def __init__(self, training_size, sequence_size):
        # class to generate data for one training set
        self.training_size = training_size
        self.sequence_size = sequence_size
        self.observations = []
        self.x_t = 2

    def init_state(self):
        self.x_t = 2

    def generate_data(self):
        for n in range(self.training_size):
            instance = []
            for i in range(self.sequence_size):
                obs = []
                self.init_state()
                while self.x_t != 5 and self.x_t != -1:
                    obs.append(self.convert_to_observation(self.x_t))
                    increment = random.choice([-1, 1])
                    self.x_t += increment
                instance.append(obs)
            self.observations.append(instance)

    def convert_to_observation(self, ind):
        obs = [0, 0, 0, 0, 0]
        obs[ind] = 1
        return obs

