import numpy as np
import networkx as nx
import random
import pandas as pd


def main():
    # 1. user can never listen to some artist
    #    then use check if by iter
    # 2. user can buy only at the next iter
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


if __name__ == '__main__':
    main()
