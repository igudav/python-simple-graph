import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from tqdm.auto import trange
from typing import Hashable, Any, Union, Optional


class Graph:
    """Class that represents the graph"""

    def __init__(self) -> None:
        self._adjacency_matrix = np.empty((0, 0), dtype=bool)
        self._key_to_idx = {}
        self._idx_to_key = {}
        self._node_payloads = {}

    def __len__(self) -> int:
        return len(self._key_to_idx)

    def add_node(self, key: Hashable, payload: Any) -> None:
        """Adds a node indexed by a key with any payload to the graph

        Args:
            key (Hashable): for accessing the node by key
            payload (Any): any payload associated with the node
        """

        if key in self._key_to_idx:
            raise KeyError(
                "The node with this payload has been already added"
            )

        self._node_payloads[key] = payload

        new_idx = self._adjacency_matrix.shape[0]
        self._key_to_idx[key] = new_idx
        self._idx_to_key[new_idx] = key

        self._adjacency_matrix = np.append(
            self._adjacency_matrix,
            np.full((1, self._adjacency_matrix.shape[0]), False),
            0
        )
        self._adjacency_matrix = np.append(
            self._adjacency_matrix,
            np.full((self._adjacency_matrix.shape[0], 1), False),
            1
        )

    def add_edge(self, key1: Hashable, key2: Hashable) -> None:
        """Adds an undirected connection between nodes

        Args:
            key1 (Hashable): the key of the first node in the graph
            key2 (Hashable): the key of the second node in the graph
        """

        if key1 not in self._key_to_idx:
            raise KeyError("There is no node with the corresponding key: "
                           f"{key1}")
        if key2 not in self._key_to_idx:
            raise KeyError("There is no node with the corresponding key: "
                           f"{key2}")

        idx1 = self._key_to_idx[key1]
        idx2 = self._key_to_idx[key2]
        self._adjacency_matrix[(idx1, idx2), (idx2, idx1)] = True

    def plot(self, ax: Optional[plt.Axes] = None):
        """Plots the graph using matplotlib

        Args:
            ax: Axes object from matplotlib where to plot the graph.
                If not provided the figure is created and shown.
        """

        show = False
        if ax is None:
            show = True
            _, ax = plt.subplots()

        angle = 2 * np.pi * np.arange(0, len(self)) / len(self)
        x = np.cos(angle)
        y = np.sin(angle)
        ax.scatter(x, y)
        for key, i in self._key_to_idx.items():
            ax.annotate(str(key), (x[i], y[i]))

        for i in range(len(self) - 1):
            for j in range(i + 1, len(self)):
                if self._adjacency_matrix[i, j]:
                    ax.plot([x[i], x[j]], [y[i], y[j]], c="C1")

        if show:
            plt.show()

    def gather_party(self) -> None:
        """Computes maximum subset of friends that does not corrupt the party.

        Nodes in graph represent people. If 2 persons are connected in the
        graph, they hate each other and the party will b corrupted if they come
        together. This function finds maximal subset of people that cannot
        corrupt the party. The algorithm is brute force. Maximal number of
        friends is printed on stdout along with associated node keys.
        """

        max_friends = 0
        friends = np.full(len(self), False)
        for subset in trange(1, 2 ** len(self)):
            mask = np.array(list(np.binary_repr(subset, width=len(self)))) == '1'
            if mask.sum() <= max_friends:
                continue

            if not self._adjacency_matrix[mask][:, mask].any():
                max_friends = mask.sum()
                friends = mask

        friends_keys = [str(self._idx_to_key[i]) for i in np.where(friends)[0]]
        if max_friends > 0:
            print(f"It is possible to gather {max_friends} friends. "
                  f"They are {', '.join(friends_keys)}")
        else:
            print("No way to gather the party :(")

    @classmethod
    def generate_random(cls,
                        n_nodes: Union[int, sps.distributions.rv_discrete],
                        edge_proba: float) -> "Graph":
        """Creates random graph

        Generate random graph using Erdos-Renyi model with optionally random
        number of nodes. The keys for the nodes are successive integers,
        payloads are empty.

        Args:
            n_nodes (int or distribution): number of nodes to generate.
                Can be a non-negative integer or scipy discrete distribution.
                The lower bound of the distribution must be non-negative.
                All parameters must be pre-specified (frozen distribution).
            edge_proba (float): probability of connection between any 2 nodes.
                Must satisfy 0 <= edge_proba <= 1.

        Returns:
            Graph
        """

        if isinstance(n_nodes, int):
            assert n_nodes >= 0, "Number of nodes must be non-negative"
            actual_n_nodes = n_nodes
        elif isinstance(n_nodes, sps.distributions.rv_frozen)\
                or isinstance(n_nodes.dist, sps.distributions.rv_discrete):
            assert n_nodes.a >= 0, "The lower bound of the distribution "\
                "must be non-negative"
            actual_n_nodes = n_nodes.rvs()
        else:
            raise TypeError("Unsupported type of n_nodes")

        assert 0 <= edge_proba <= 1

        g = Graph()
        for i in range(actual_n_nodes):
            g.add_node(i, None)

        for i in range(actual_n_nodes - 1):
            for j in range(i + 1, actual_n_nodes):
                if np.random.rand() < edge_proba:
                    g.add_edge(i, j)

        return g

