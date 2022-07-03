from graph import Graph

import matplotlib.pyplot as plt
import scipy.stats as sps


def graph_demo():
    NROWS = 2
    NCOLS = 3

    _, axs = plt.subplots(nrows=NROWS, ncols=NCOLS)

    for i in range(NROWS):
        for j in range(NCOLS):
            g = Graph.generate_random(n_nodes=sps.distributions.poisson(10),
                                      edge_proba=.3)
            g.plot(ax=axs[i][j])
            print(f"Subplot {(i, j)}")
            g.gather_party()
            print()

    plt.show()


if __name__ == "__main__":
    graph_demo()
