# -*- coding: utf-8 -*-
import numpy as np


def get_data(file_dir):
    select_dict = {}
    raw_data = []
    user_list = []
    url_list = []
    user_seen = set(user_list)
    url_seen = set(url_list)
    with open(file_dir, 'r') as f_data:
        next(f_data)
        for line in f_data:
            temp_list = line.split('\t')
            user = temp_list[0]
            select_url = temp_list[1]
            if user not in user_seen:
                user_seen.add(user)
                user_list.append(user)
            if select_url not in url_seen:
                url_seen.add(select_url)
                url_list.append(select_url)
            # check user key
            if user in select_dict:
                # add booked url to user list
                if select_url not in select_dict[user]:
                    select_dict[user].append(select_url)
            # add new user
            else:
                select_dict[user] = []
    return user_list, url_list, select_dict


def logging_policy(user_list, url_list, select_dict, length_l):
    event_s = []
    # create L length events
    for i in range(length_l):
        one_event_dict = {}
        # all uniform random select
        # choice user
        user = np.random.choice(user_list)
        # choice k=20 arms
        arms_pool = np.random.choice(url_list, 20)
        # pull a arm from items
        arm = np.random.choice(arms_pool)
        # get reward
        reward_seen = set(select_dict[user])
        if arm in reward_seen:
            reward = 1
        else:
            reward = 0
        # optimal reward
        for arm_r in arms_pool:
            if arm_r in reward_seen:
                optimal_rewrad = 1
                break
            else:
                optimal_rewrad = 0
        # record this event the form:[x,a,ra,items]
        one_event_dict['x'] = user
        one_event_dict['a'] = arm
        one_event_dict['ra'] = reward
        one_event_dict['items'] = arms_pool
        one_event_dict['op_ra'] = optimal_rewrad

        # add a event to list
        event_s.append(one_event_dict)
    return event_s


class DeliciousData:

    def __init__(self):
        self.user, self.url_pool, self.booked_list = get_data(
            '../data/hetrec2011-delicious-2k/user_taggedbookmarks.dat')

    def get_logging_events(self, L_length):
        events = logging_policy(self.user, self.url_pool,
                                self.booked_list, L_length)
        return events
