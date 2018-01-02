# -*- coding: utf-8 -*-
from random_model import RandomModel
import matplotlib.pyplot as plt
from get_event_delicious import DeliciousData


# length of stream
length_l = 1000

# plot line color
colors = ['r']


def initialize_models():
    '''Initialize specific models to be evaluated'''

    models = []
    models.append(RandomModel())
    return models


if __name__ == '__main__':
    delicious_data = DeliciousData()
    stream = delicious_data.get_logging_events(length_l)
    models = initialize_models()

    # model evaluator
    print('Model evaluator')
    for model in models:
        count = 0
        print(model.name)
        for event in stream:
            count += 1
            if count % 50 == 0:
                print('Stream ', count)

            # model choose an action
            choice = model.take_action(event)

            # judge if model choose the same arm as logging policy
            if choice == event['a']:
                # construct labeled data
                feature = [int(event['x']), int(event['a'])]
                reward = int(event['ra'])
                reward_optimal = int(event['op_ra'])

                # update model
                model.update(feature, reward, reward_optimal)

    # show regret
    for model, color in zip(models, colors):
        name, step, rewards, rewards_optimal = model.get_info()

        # calculate cumulative regret
        cumulative_regrets = []
        for s in range(step):
            cr = rewards_optimal[s] - rewards[s]
            if len(cumulative_regrets) == 0:
                cumulative_regrets.append(cr)
            else:
                cumulative_regrets.append(cr + cumulative_regrets[-1])

        # plot this model
        x = range(1, step + 1)
        plt.plot(x, cumulative_regrets, label=name, linewidth=3,
                 color=color)

    plt.xlabel('Step')
    plt.ylabel('Cumulative regret')
    plt.title('Cumulative regret on offline evaluator')
    plt.legend()
    plt.show()
