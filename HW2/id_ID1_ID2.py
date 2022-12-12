class BiddingAgent:
    def __init__(self):
        pass

    def get_bid(self, num_of_agents, P, q, v):
        """"
        :param num_of_agents: number of agents competing in this round
        :param P:P_1,...P_n, where P_i is the probability of a user clicking on position i (n <= num_agents)
        :param q: quality score
        :param v: value in case of click
        :return:
        """
        res = (v / 4)
        return res

    def notify_outcome(self, reward, outcome, position):
        """
        :param reward: The auction's reward
        :param outcome: The auction's payment
        :param position: The position obtained at the auction
        :return: We won't use its return value
        """
        # TODO decide what/if/when to store. Could be used for future use
        pass

    def get_id(self):
        """
        :return: name of this file
        """
        return "id_206320772_313510679"
