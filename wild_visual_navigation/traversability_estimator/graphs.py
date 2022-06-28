from wild_visual_navigation import WVN_ROOT_DIR
import os
from os.path import join
import networkx as nx
import torch
from threading import Lock


class BaseGraph:
    def __init__(self):
        """Initializes the graph"""

        # Initialize graph
        self.graph = nx.Graph()
        self.last_added_node = None

        # Mutex
        self.lock = Lock()

    def __str__(self):
        return str(self.graph)

    def add_node(self, state):
        """Adds a node to the graph and creates edge to the latest node

        Returns:
            model (type): Description
        """
        # Add node
        with self.lock:
            self.graph.add_node(state, timestamp=state.get_timestamp())

            # Add edge to latest
            if self.last_added_node is not None:
                self.graph.add_edge(state, self.last_added_node, distance=state.distance_to(self.last_added_node))
            else:
                self.first_node = state

        # Update last added node
        self.last_added_node = state

    def add_edge(self, state1, state2):
        with self.lock:
            self.graph.add_edge(state1, state2, distance=state1.distance_to(state2))

    def clear(self):
        with self.lock:
            self.graph.clear()

    def get_first_node(self):
        if self.get_num_nodes() > 0:
            return self.get_nodes()[0]

    def get_last_node(self):
        if self.get_num_nodes() > 0:
            return self.get_nodes()[-1]

    def get_num_nodes(self):
        with self.lock:
            return len(self.graph.nodes)

    def get_num_edges(self):
        with self.lock:
            return len(self.graph.edges)

    def get_nodes(self):
        with self.lock:
            nodes = sorted(self.graph.nodes)
        return nodes

    def get_node_with_timestamp(self, timestamp, eps=1e-12):
        def approximate_timestamp_filter(node):
            return abs(node.get_timestamp() - timestamp) < eps

        with self.lock:
            nodes = sorted(nx.subgraph_view(self.graph, filter_node=approximate_timestamp_filter).nodes)
        return nodes

    def get_nodes_within_radius(self, node, radius):
        # Find closest node in the graph. This is useful when we are finding nodes correrponding to another graph
        closest_nodes = self.get_node_with_timestamp(node.get_timestamp(), eps=1e-2)

        nodes = []
        try:
            closest_node = closest_nodes[0]
            with self.lock:
                length, path = nx.single_source_dijkstra(self.graph, closest_node, cutoff=radius, weight="distance")
            nodes = list(length)[1:]  # first node is the query node
        except Exception as e:
            pass
        return nodes

    def get_nodes_within_timespan(self, t_ini, t_end, open_interval=False):
        """Returns all nodes in (t_ini, t_end)

        Returns:
            model (type): Description
        """

        def temporal_filter(node):
            if open_interval:
                return node.get_timestamp() > t_ini and node.get_timestamp() < t_end
            else:
                return node.get_timestamp() >= t_ini and node.get_timestamp() <= t_end

        with self.lock:
            nodes = list(nx.subgraph_view(self.graph, filter_node=temporal_filter).nodes)
        return nodes

    def remove_nodes(self, nodes):
        with self.lock:
            self.graph.remove_nodes_from(nodes)

    def remove_nodes_within_radius(self, node, radius):
        nodes_to_remove = self.get_nodes_within_radius(node, radius)
        self.remove_nodes(nodes_to_remove)

    def remove_nodes_within_timestamp(self, t_ini, t_end):
        nodes_to_remove = self.get_nodes_within_timespan(t_ini, t_end, open_interval=False)
        self.remove_nodes(nodes_to_remove)


class GlobalGraph(BaseGraph):
    def __init__(self):
        super().__init__()


class LocalGraph(BaseGraph):
    def __init__(self, time_window):
        super().__init__()
        self.time_window = time_window

    def add_node(self, state):
        """Adds a node to the graph and removes old nodes"""
        # Add node
        super().add_node(state)

        # Remove all nodes from the beginning of time till right before the time window
        t_end = state.get_timestamp() - self.time_window
        self.remove_nodes_within_timestamp(0, t_end)

    def remove_old_nodes(state):
        """Adds a node to the graph and removes old nodes"""
        # Remove all nodes from the beginning of time till right before the time window
        t_end = self.get_last_node().get_timestamp() - self.time_window
        self.remove_nodes_within_timestamp(0, t_end)


def run_base_graph():
    from wild_visual_navigation.traversability_estimator import BaseNode
    import torch
    from liegroups.torch import SO3, SE3
    import matplotlib.pyplot as plt

    # Create graph
    graph = BaseGraph()
    N = 10

    nodes_list = []
    for i in range(N):
        t = i
        s = BaseNode(timestamp=i, T_WB=SE3(SO3.identity(), torch.Tensor([i / 10.0, 0, 0])).as_matrix())
        nodes_list.append(s)
        graph.add_node(s)

    # Check graph as list
    assert nodes_list == graph.get_all_nodes()

    # Check number of nodes
    assert graph.get_num_nodes() == N

    # Check number of edges
    assert graph.get_num_edges() == N - 1

    # Get nodes within radius
    radius = 0.2
    query_node = graph.get_node_with_timestamp(5.0)[0]
    nodes = graph.get_nodes_within_radius(query_node, radius)

    # Get nodes within timespan
    nodes = graph.get_nodes_within_timespan(0.0, 3.0, open_interval=True)
    assert len(nodes) == 2

    # Get nodes within timespan
    nodes = graph.get_nodes_within_timespan(0.0, 3.0, open_interval=False)
    assert len(nodes) == 4

    # Remove nodes
    graph.remove_nodes(nodes)
    assert graph.get_num_nodes() == 6


def run_local_graph():
    from wild_visual_navigation.traversability_estimator import BaseNode
    import torch
    from liegroups.torch import SO3, SE3
    import matplotlib.pyplot as plt

    # Create graph
    W = 5
    N = 20
    graph = LocalGraph(time_window=W)

    nodes_list = []
    for i in range(N):
        t = i
        s = BaseNode(timestamp=t, T_WB=SE3(SO3.identity(), torch.Tensor([i / 10.0, 0, 0])).as_matrix())
        nodes_list.append(s)
        graph.add_node(s)
        assert graph.get_first_node().get_timestamp() >= t - W


if __name__ == "__main__":
    run_base_graph()
    run_local_graph()
