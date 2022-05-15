import numpy as np
import networkx as nx
import random
import pandas as pd


def main():
    # random.seed(772)
    G = build_graph()
    artists = [989, 16326, 511147, 532992]
    # artist: 989, influencers: [601148, 798855, 265474, 655767, 411093], amount: 1724
    # list_ = simulation_per_artist(G, [548221,411093,874459,175764,852394], 989, [])
    # print(len(list_))
    # return
    # simulate_influencers_influence(G, artists)  # this if for us to study the data
    sourceFile = open('sim_res.txt', 'w')
    for artist in artists:
        print(f'artist: {artist}')
        influencers, amount = hill_climb(G, artist)
        print(f'artist: {artist}, influencers: {influencers}, amount: {amount}')
        print(f'artist: {artist}, influencers: {influencers}, amount: {amount}', file=sourceFile)
    sourceFile.close()
    return
    # prob_hist = calc_graph_prob()  # TODO enable
    prob_hist = []
    total_infected = 0
    for artist in artists:
        influencers = find_most_influences(G)
        print(f'artist: {artist}')
        infected_list = simulation_per_artist(G, influencers, artist, prob_hist)
        print(f"Total user infected for artist {artist}: {len(infected_list)}")
        total_infected += len(infected_list)
    print(f'Total infected: {total_infected}')
    pass


def hill_climb(G, artist):
    influencers = []
    amount = 0
    while len(influencers) < 5:
        IC = {}
        for node in G.nodes() - influencers:
            amount_infected = len(simulation_per_artist(G, influencers + [node], artist, []))
            IC[node] = amount_infected
        best_influencer = max(IC, key=IC.get)
        influencers.append(best_influencer)
        amount = IC[best_influencer]
        print(f'influencers: {influencers}, amount: {IC[best_influencer]}')
    return influencers, amount


def calc_potential_potential(G, node, artist, influencers):
    potential_influence = 0
    for neighbor in nx.neighbors(G, node):
        # potential_influence +=  # TODO
        pass
    return potential_influence


def simulate_influencers_influence(G: nx.Graph, artists: list):
    prob_hist = []
    k = 10
    for artist in artists:
        influencers_lists, added = find_most_k_influences(G, k)
        max_influence = 0
        max_influencers = []
        for influencers in influencers_lists:
            total_infected = 0
            infected_list = simulation_per_artist(G, list(influencers), artist, prob_hist)
            total_infected += len(infected_list)
            if max_influence < total_infected:
                max_influence = total_infected
                max_influencers = influencers
            print(f'influencers: {influencers}, k = {k}, ', end='')
            print(f'Total infected: {total_infected}, artist: {artist}')
        print(f'max influence: {max_influence}, influencers: {max_influencers}')
    pass


def simulation_per_artist(G: nx.Graph, influencers: list, artist, prob_hist):
    infected_list = influencers
    for t in range(1, 7):
        now_infected = []
        total_adj = 0
        for infected in infected_list:
            num_adj = 0
            num_infected = 0
            for adj in G.neighbors(infected):
                total_adj += 1
                num_adj += 1
                if adj not in infected_list and adj not in now_infected:
                    p = random.uniform(0, 1)
                    if p < calc_probability_to_infect(G, adj, infected_list, artist):
                        num_infected += 1
                        now_infected.append(adj)
        # print(f"Iter: {t}, infected: {len(now_infected)}, from {len(infected_list)}"
        #       f", out of: {total_adj}, ratio: {round(len(infected_list)/total_adj, 3)}")
        infected_list = infected_list + now_infected
        # G = update_graph(G, prob_hist)  # TODO enable this
    return infected_list


def find_most_influences(G: nx.Graph):
    max5 = []
    while len(max5) < 5:
        max_deg = 0
        max_node = None
        for node in np.array(G.degree):
            if max_deg < node[1] and node[0] not in max5:
                max_deg = node[1]
                max_node = node[0]
        max5.append(max_node)
    return max5


def find_most_k_influences(G: nx.Graph, k: int):
    max_k = []
    while len(max_k) < k:
        max_deg = 0
        max_node = None
        for node in np.array(G.degree):
            if max_deg < node[1] and node[0] not in max_k:
                max_deg = node[1]
                max_node = node[0]
        max_k.append(max_node)

    # added = []
    # max5 = []
    # while len(max5) < 5:
    #     random_index = random.randint(0, len(max_k) - 1)
    #     if random_index not in added:
    #         added.append(random_index)
    #         max5.append(max_k[random_index])
    # print(f'{max5}, k = {k}')
    # return max5, added

    import itertools
    max5 = []
    max5 = itertools.combinations(max_k, 5)
    # print(list(max5))
    return max5, []


def calc_graph_prob():
    G0, G1 = graph_it('./data/instaglam_1.csv'), graph_it('./data/instaglam0.csv')
    num_nodes = G0.number_of_nodes()
    pairs = np.zeros(num_nodes)
    connected_pairs = np.zeros(num_nodes)
    visit = []
    print("Calculating probability vector...")
    for node in G0.nodes():
        visit.append(node)
        for other in G0.nodes() - visit:
            k = len(list(nx.common_neighbors(G0, node, other)))
            if not G0.has_edge(node, other):  # TODO if k in prob_hist.keys() else p = ?
                pairs[k] += 1
                if G1.has_edge(node, other):
                    connected_pairs[k] += 1
    prob_hist = np.zeros(len(pairs))
    for i in range(len(pairs)):
        if pairs[i] != 0:
            prob_hist[i] = (connected_pairs[i] / pairs[i])
    return prob_hist


def graph_it(path: str):
    df = pd.read_csv(path)
    data = df.to_dict(orient='list')
    users = np.array(data['userID'])
    friends = np.array(data['friendID'])
    G = nx.Graph()
    G.add_nodes_from(np.unique(np.append(users, friends)))
    G.add_edges_from(list(zip(users, friends)))
    return G


def build_graph():
    path = './data/instaglam0.csv'
    G = graph_it(path)
    df = pd.read_csv('./data/spotifly.csv')
    spotifly_dict = df.groupby('userID').apply(lambda x: dict(zip(x[' artistID'], x['#plays']))).to_dict()
    nx.set_node_attributes(G, spotifly_dict)
    return G


def update_graph(G, prob_hist):
    print("Updating graph...")
    visit = []
    for node in G.nodes:
        visit.append(node)
        for other in G.nodes - visit:
            if not G.has_edge(node, other):  # TODO check e=(u,v), e=(v,u)
                k = len(list(nx.common_neighbors(G, node, other)))
                prob = prob_hist[k]  # TODO check for index out of bound error
                p = random.uniform(0, 1)
                if p < prob:
                    G.add_edge(node, other)
            pass
        pass
    return G


# probability to infect from neighbor
def calc_probability_to_infect(G: nx.Graph, node, infected, artist):
    neighbors = G.neighbors(node)
    N_t, B_t = 0, 0
    for neighbor in neighbors:
        N_t += 1
        if neighbor in infected:
            B_t += 1
    if artist not in G.nodes[node].keys():
        h = 0
    else:
        h = G.nodes[node][artist]
    if h != 0:
        return (h * B_t) / (1000 * N_t)
    else:
        return B_t / N_t
    pass


if __name__ == '__main__':
    main()
