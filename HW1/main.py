import networkx
import numpy as np
import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt  # TODO remove

# TODO remove
""" 
working plane:

what we have:

what we need:
needs to calculate the chance to add a new node
    iterate over each uninfected user and infect it by the probability
    then check for 
"""


def main():
    # build graphs from the 2 csv files
    G0 = build_graph('./data/instaglam0.csv')
    G1 = build_graph('./data/instaglam_1.csv')
    p = calc_graph_prob(G1, G0)
    G_diff = G0

    # get diff graph
    for edge in np.array(G1.edges):
        G_diff.remove_edge(edge[0], edge[1])

    # calculate graph adding edge prob
    # histogram
    hist = get_degree_histogram(G1)
    plt.hist(hist, 10)
    plt.show()

    #
    pass


def run(influencers):
    infected_list = influencers
    for t in range(1, 7):
        now_infected = []
        for infected in infected_list:
            for adj in infected.adjency:
                p = random.uniform(0, 1)
                if p < adj.probability:  # TODO calculate probability
                    now_infected.append(adj)
        infected_list = infected_list + now_infected
        update_graph()
    pass


def function(G1, G_diff):
    hist = {}
    for edge in G_diff.edges:
        # adj1 = nx.read_adjlist(edge[0]
        # adj12 = edge[1]
        pass
    print(G_diff.adjacency())
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


# TODO implement find probability to add new edge
def calc_graph_prob(G1, G0):
    # TODO find newly added edges at iter 0
    # TODO get the ratio {newly_added} / {sum_edges}
    # TODO return probability
    p = 0  # TODO
    return p


# TODO implement
def update_graph(G, p):
    # TODO find all pair of nodes with common friends
    #   use that probability and add new edge only if it closes a triangle
    #   2. add a edge if random < p
    pass


def get_probability(G, node, infected, artist):
    # TODO find how many adj
    # TODO find how many infected adj
    # TODO find h
    return calc_probability(N_t, B_t, h)


# TODO make function or integrate in this one to get the input values
def calc_probability(N_t, B_t, h):
    """
    calculate the probability to buy product
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


def get_degree_histogram(graph):
    hist = {}
    for d in graph.degree():
        if d[1] in hist.keys():
            hist[d[1]] += 1
        else:
            hist[d[1]] = 1
    # plt.hist(histogram)
    # plt.show()
    return hist


if __name__ == '__main__':
    main()


