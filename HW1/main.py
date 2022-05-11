import networkx
import numpy as np
import networkx as nx
import random
# import matplotlib.pyplot as plt  # TODO remove
import pandas as pd


def main():
    # build graphs from the 2 csv files
    G0 = build_graph('./data/instaglam0.csv')
    G1 = build_graph('./data/instaglam_1.csv')
    p = calc_graph_prob(G1, G0)

    df = pd.read_csv('./data/spotifly.csv')
    spotifly = df.to_dict(orient='list')

    # histogram
    hist = get_degree_histogram(G1)

    # probability to add new edge if they have common friends
    prob = calc_graph_prob(G1, G0)

    influences = [197117, 563940, 385510, 865448, 381409]  # top 5 most degree TODO only temporary to run spread func
    run(G0, influences, spotifly, prob)
    pass


def find_most_influences():
    # max_deg = 0
    # max_node = 0
    # for edge in np.array(G0.degree):
    #     if max_deg < edge[1] and edge[0] not in [197117, 563940, 385510, 865448, 381409]:
    #         max_node = edge[0]
    #         max_deg = edge[1]
    # print(max_node, max_deg)
    # max node at 0 [197117, 21], [563940, 20], [385510, 18], [865448, 18], [381409, 16]
    # most degrees at G0: [197117, 563940, 385510, 865448, 381409]
    pass


def run(G, influences, spotifly, prob):
    infected_list = influences
    for t in range(1, 7):
        now_infected = []
        for infected in infected_list:
            # artist = infected.get_artist()  # TODO how to find artist
            artist = None
            for adj in G.neighbors(infected):
                p = random.uniform(0, 1)
                if p < calc_probability(G, adj, infected_list, spotifly, artist):
                    now_infected.append(adj)
        infected_list = infected_list + now_infected
        update_graph(G, prob)
    pass


def calc_graph_prob(G1, G0):
    G_diff = G0
    for edge in np.array(G1.edges):
        G_diff.remove_edge(edge[0], edge[1])
    p = len(G_diff.edges) / len(G1.edges)
    return p


def build_graph(path):
    df = pd.read_csv(path)
    data = df.to_dict(orient='list')
    users = np.array(data['userID'])
    friends = np.array(data['friendID'])

    G = nx.Graph()
    G.add_nodes_from(np.unique(np.append(users, friends)))
    G.add_edges_from(list(zip(users, friends)))
    return G


def update_graph(G, prob):
    for node in G.nodes:
        for other in G.nodes:
            # check if they have common friend\s TODO make it work for k number of friends
            if G.neighbors(node) == G.neighbors(other):  # TODO change the condition to find cut between lists
                p = random.uniform(0, 1)
                if p < prob:
                    G.add_edge(node, other)
    return G


# probability to infect from neighbor
def calc_probability(G, node, infected, spotifly, artist):
    neighbors = G.neighbors(node)
    N_t = len(neighbors)
    B_t = 0
    for neighbor in neighbors:
        if neighbor in infected:
            B_t += 1
    h = 0
    for record in spotifly:
        if record[0] == node and record[1] == artist:
            h = record[2]
            break
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


