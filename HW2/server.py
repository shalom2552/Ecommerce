import time
import random

import numpy as np
import pandas as pd
import CONSTANTS
from tqdm import tqdm

import os


def cap_prob(p):
    if p > 1.0:
        return 1.0
    elif p < 0.00001:
        return 0.00001
    else:
        return p


def get_agent_list():
    """
    In our simulations, we will read your files. Here
    we have several dummy files and your agent. Feel free to implement many agents and test them against one another
    :return: list of all agents competing in the game
    """
    file_list = [f.rstrip(".py") for f in os.listdir('.') if (os.path.isfile(f) and f.startswith("id_"))]
    agent_list = []
    for i, f in enumerate(file_list):
        print(i, f)
        mod = __import__(f)
        try:
            agent = mod.BiddingAgent()
            if is_agent_valid(agent):
                agent_list.append(agent)
            else:
                print("file {0} introduced an invalid agent".format(f))
        except Exception as e:
            print(e)
            print("file {0} raised an exception".format(f))
    return agent_list


def is_agent_valid(ba):
    # has the properties
    if not hasattr(ba, 'get_bid') or not hasattr(ba, 'get_id') or not hasattr(ba, 'notify_outcome'):
        return False
    return True


class BookKeeping:
    def __init__(self, agent):
        self.id = agent.get_id()
        self.played_rounds = 0
        self.sum_payments = 0.
        self.sum_rewards = 0.
        self.sum_computation_time = 0.
        self.time_out = False

    def add_comp_time(self, t):
        self.sum_computation_time += t

    def set_time_out(self):
        self.time_out = True

    def get_avg_comp_time(self):
        if self.played_rounds == 0:
            return 0
        return self.sum_computation_time / self.played_rounds

    def update(self, reward, payment):
        self.played_rounds += 1
        self.sum_payments += payment
        self.sum_rewards += reward

    def get_id(self):
        return self.id

    def get_all(self):
        return {"id": self.id, "played_rounds": self.played_rounds, "sum_payments": self.sum_payments,
                "sum_rewards": self.sum_rewards, "sum_computation_time": self.sum_computation_time,
                "sum_utility": self.sum_rewards - self.sum_payments,
                "avg_utility": (self.sum_rewards - self.sum_payments) / self.played_rounds,
                "was_timeout": self.time_out}


def sample_relevant_agents(all_agents):
    """
    Samples roughly 10% of the agents. If thats
    """
    cap = len(all_agents) / (CONSTANTS.DEFAULT_NUM_AGENTS * 100)
    relevant = []
    for agent in all_agents:
        if random.random() < cap:
            relevant.append(agent)
    if len(relevant) == 0:
        relevant = random.sample(all_agents, k=CONSTANTS.MIN_NUM_AGENTS)
    random.shuffle(relevant)
    return relevant


def get_timeout_agents(relevant_agents, book_dict):
    lst = []
    for agent in relevant_agents:
        book = book_dict[agent]
        if book.get_avg_comp_time() > CONSTANTS.TIME_CAP:
            print("Agent {0} was removed due to slow operation".format(agent.get_id()))
            lst.append(agent)
            book.set_time_out()
    return lst


def sample_variables(num_agents):
    v_list = [max(np.random.normal(CONSTANTS.R_MEAN, CONSTANTS.R_STD), 0) for _ in range(num_agents)]
    q_dist = np.random.choice(range(len(CONSTANTS.Q_TYPE)), size=num_agents, p=CONSTANTS.Q_TYPE)
    q_list = [cap_prob(np.random.normal(*CONSTANTS.RELEVANCE_MEAN_STD[z])) for z in q_dist]

    prob_click = []
    prob_first = cap_prob(np.random.normal(*CONSTANTS.P_CLICK))
    decay = cap_prob(np.random.normal(*CONSTANTS.D_DECAY))
    pv = prob_first
    while pv >= CONSTANTS.MIN_PCLICK:
        prob_click.append(pv)
        pv = pv * decay

    return v_list, q_list, prob_click


def gsp(bids, q_list, n_positions):
    """
    :return: dict, where k is an agent index and v=(winning position, payment)
    """
    gsp_outcome = {}
    bids_q = [(bids[i][0], bids[i][1] * q_list[i]) for i in
              range(len(q_list))]  # like we've seen in class, b^* = (b_i,q_i)
    sorted_bids = sorted(bids_q, key=lambda b: b[1], reverse=True)
    sorted_bids.append((len(bids), CONSTANTS.MIN_PAYMENT))
    for pos in range(min(n_positions, len(bids))):
        agent_index = sorted_bids[pos][0]
        gsp_outcome[agent_index] = (pos, sorted_bids[pos + 1][1])
    return gsp_outcome


def sample_rewards(gsp_outcome, q_list, v_list, prob_click):
    # for each agent: from the position, get p*q
    # sample Bernoulli
    rewards = {}
    for k, v in gsp_outcome.items():  # k is a name of an agent, v is the position she got
        rewards[k] = 0.
        if random.random() < q_list[k] * prob_click[v[0]]:
            rewards[k] = v_list[k]
    return rewards


def sequential_game():
    all_agents = get_agent_list()
    book_dict = {a: BookKeeping(a) for a in all_agents}
    for _ in tqdm(range(CONSTANTS.NUM_ROUNDS)):
        relevant_agents = sample_relevant_agents(all_agents)
        num_of_agents = len(relevant_agents)
        v_list, q_list, prob_click = sample_variables(num_of_agents)
        bids = []
        for i, agent in enumerate(relevant_agents):
            comp_t = time.time()
            bids.append((i, agent.get_bid(num_of_agents, prob_click, q_list[i], v_list[i])))
            book_dict[agent].add_comp_time(time.time() - comp_t)

        gsp_outcome = gsp(bids, q_list, len(prob_click))
        rewards = sample_rewards(gsp_outcome, q_list, v_list, prob_click)

        for i, agent in enumerate(relevant_agents):
            comp_t = time.time()
            reward, payment, position = 0., 0., -1
            if i in rewards:
                reward, payment, position = rewards[i], gsp_outcome[i][1], gsp_outcome[i][0]
            agent.notify_outcome(reward, payment, position)
            book_dict[agent].add_comp_time(time.time() - comp_t)
            book_dict[agent].update(reward, payment)

        # Agents that were super slow are removed
        to_remove = get_timeout_agents(relevant_agents, book_dict)
        for agent in to_remove:
            all_agents.remove(agent)

    return book_dict


def main():
    book_dict = sequential_game()
    lst = [bk.get_all() for bk in book_dict.values()]
    df = pd.DataFrame(lst)
    print(df)


if __name__ == '__main__':
    main()
