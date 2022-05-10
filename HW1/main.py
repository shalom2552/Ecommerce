import numpy as np
import networkx as nx
import random
import pandas as pd


def main():
    G0 = build_graph('./data/instaglam0.csv')
    G1 = build_graph('./data/instaglam_1.csv')
    Gtemp = G0

    for edge in np.array(G1.edges):
        Gtemp.remove_edge(edge[0], edge[1])
    print((np.array(Gtemp.edges)))
    pass


def build_graph(path):
    df = pd.read_csv(path)
    data = df.to_dict(orient='list')
    users = np.array(data['userID'])
    friends = np.array(data['friendID'])

    G = nx.Graph()
    G.add_nodes_from(np.unique(np.append(users, friends)))
    G.add_edges_from(list(zip(users, friends)))
    return G


def decision_rule():

    pass


def calc_probability(N_t, B_t, h):
    """
    calculater the probability to buy product
    :param N_t: number of neighbors of the user at time t
    :param B_t: number of neighbors who bought the product by time t
    :param h: number of times the user listened to the artist
    :return: probability for the user to buy product at time t+1
    """
    if h != 0:
        return h * B_t / N_t * 1000  # (h * B_t) / (1000 * N_t)
    else:
        return B_t / N_t
    pass

# 1. calculate by instagram the probability to make new friends TODO
# 2. each step run for each user the probability to buy and after update the graph
#       the probability will be calculated againg for each user after each iteration
# 3.


# 1. user can never listen to some artist TODO
#    then use check if by iter
# 2. user can buy only at the next iter


if __name__ == '__main__':
    main()


