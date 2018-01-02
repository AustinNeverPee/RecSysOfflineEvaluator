# -*- coding: utf-8 -*-

import random


class RandomModel:
    '''Model that takes random actions'''

    def __init__(self):
        self.name = 'Random Model'
        self.step = 0
        self.rewards = []
        self.rewards_optimal = []

    def update(self, feature, reward, reward_optimal):
        '''Update model with actual reward'''

        self.step += 1
        self.rewards.append(reward)
        self.rewards_optimal.append(reward_optimal)

    def take_action(self, context):
        '''Take action according to model prediction'''

        return random.choice(context['items'])

    def predict(self, feature):
        '''Predict reward'''

        pass

    def get_info(self):
        '''Return model information'''

        return self.name, self.step, self.rewards, self.rewards_optimal
