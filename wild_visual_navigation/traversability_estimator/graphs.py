from wild_visual_navigation import WVN_ROOT_DIR
from .nodes import BaseNode
import os
from os.path import join
import networkx as nx
import random
import torch
from threading import Lock


class BaseGraph:
    def __init__(self):
        """Initializes a graph with basic functionalities

        Args:
            None

        Returns:
            A BaseGraph object
        """

        # Initialize graph
        self._graph = nx.Graph()
        self._first_node = None
        self._last_added_node = None

        # Mutex
        self._lock = Lock()

    def __str__(self):
        return str(self._graph)

    def __getstate__(self):
        """We modify the state so the object can be pickled"""
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        """We modify the state so the object can be pickled"""
        self.__dict__.update(state)
        self._lock = Lock()

    def change_device(self, device):
        for n in self._graph.nodes:
            n.change_device(device)

    def add_node(self, node: BaseNode):
        """Adds a node to the graph and creates edge to the latest node

        Returns:
            model (type): Description
        """
        # Add node
        with self._lock:
            self._graph.add_node(node, timestamp=node.timestamp)

            # Add edge to latest
            if self._last_added_node is not None:
                self._graph.add_edge(node, self._last_added_node, distance=node.distance_to(self._last_added_node))
            else:
                self._first_node = node

        # Update last added node
        self._last_added_node = node
        return True

    def add_edge(self, node1: BaseNode, node2: BaseNode):
        with self._lock:
            self._graph.add_edge(node1, node2, distance=node1.distance_to(node2))
        return True

    def clear(self):
        with self._lock:
            self._graph.clear()

    def get_first_node(self):
        return self._first_node

    def get_last_node(self):
        return self._last_added_node

    def get_num_nodes(self):
        with self._lock:
            return len(self._graph.nodes)

    def get_num_valid_nodes(self):
        with self._lock:
            return sum([n.is_valid() for n in self._graph.nodes])

    def get_num_edges(self):
        with self._lock:
            return len(self._graph.edges)

    def get_nodes(self):
        with self._lock:
            nodes = sorted(self._graph.nodes)
        return nodes

    def get_valid_nodes(self):
        with self._lock:
            return sorted([n for n in self._graph.nodes if n.is_valid()])

    def get_n_random_valid_nodes(self, n=None):
        nodes = self.get_valid_nodes()
        random.shuffle(nodes)
        if n is None:
            return nodes
        else:
            return nodes[:n]

    def get_node_with_timestamp(self, timestamp: float, eps: float = 1e-12):
        def approximate_timestamp_filter(node):
            return abs(node.timestamp - timestamp) < eps

        with self._lock:
            nodes = sorted(nx.subgraph_view(self._graph, filter_node=approximate_timestamp_filter).nodes)

        return nodes[0] if len(nodes) > 0 else None

    def get_nodes_within_radius_range(
        self, node: BaseNode, min_radius: float, max_radius: float, time_eps: float = 1, metric: str = "dijkstra"
    ):
        # Find closest node in the graph (timestamp). This is useful when we are finding nodes corresponding to another graph
        closest_node = self.get_node_with_timestamp(node.timestamp, eps=time_eps)

        nodes = []
        try:
            with self._lock:
                if metric == "dijkstra":
                    length, path = nx.single_source_dijkstra(
                        self._graph, closest_node, cutoff=max_radius, weight="distance"
                    )
                    nodes = list(length)[1:]  # first node is the query node
                elif metric == "pose":

                    def pose_distance_filter(other):
                        d = abs(other.distance_to(node))
                        return d >= min_radius and d < max_radius

                    nodes = sorted(nx.subgraph_view(self._graph, filter_node=pose_distance_filter).nodes)

        except Exception as e:
            print(f"[get_nodes_within_radius_range] Exception: {e}")
        return sorted(nodes)

    def get_nodes_within_timespan(self, t_ini: float, t_end: float, open_interval: bool = False):
        """Returns all nodes in (t_ini, t_end)

        Returns:
            model (type): Description
        """

        def temporal_filter(node: BaseNode):
            if open_interval:
                return node.timestamp > t_ini and node.timestamp < t_end
            else:
                return node.timestamp >= t_ini and node.timestamp <= t_end

        with self._lock:
            nodes = list(nx.subgraph_view(self._graph, filter_node=temporal_filter).nodes)
        return nodes

    def remove_nodes(self, nodes: list):
        with self._lock:
            self._graph.remove_nodes_from(nodes)

    def remove_nodes_within_radius_range(
        self, node: BaseNode, min_radius: float = 0, max_radius: float = float("inf"), metric: str = "dijkstra"
    ):
        nodes_to_remove = self.get_nodes_within_radius_range(
            node, min_radius=min_radius, max_radius=max_radius, metric=metric
        )
        self.remove_nodes(nodes_to_remove)

    def remove_nodes_within_timestamp(self, t_ini: float, t_end: float):
        nodes_to_remove = self.get_nodes_within_timespan(t_ini, t_end, open_interval=False)
        self.remove_nodes(nodes_to_remove)


class TemporalWindowGraph(BaseGraph):
    def __init__(self, time_window: float, edge_distance: float = None):
        """Initializes a graph that keeps nodes within a time window

        Args:
            time_window (float): maximum time to keep nodes (counting from the last added node)
            edge_distance (float): threshold to avoid adding nodes that are too close

        Returns:
            A TemporalWindowGraph
        """
        super().__init__()
        self._time_window = time_window
        self._edge_distance = edge_distance

    def add_node(self, node: BaseNode):
        """Adds a node to the graph and removes old nodes"""
        if self._edge_distance is not None and self.get_last_node() is not None:
            # compute distance to last node and do not add the node if it's too close
            d = node.distance_to(self.get_last_node())
            if d < self._edge_distance:
                return False

        # Add node
        out = super().add_node(node)

        # Remove all nodes from the beginning of time till right before the time window
        t_end = node.timestamp - self._time_window
        self.remove_nodes_within_timestamp(0, t_end)
        return out


class DistanceWindowGraph(BaseGraph):
    def __init__(self, max_distance: float, edge_distance: float = None):
        super().__init__()
        """Initializes a graph that keeps nodes within a max distance

        Args:
            max_distance (float): maximum distance to keep nodes (measured from last added node)
            edge_distance (float): threshold to avoid adding nodes that are too close

        Returns:
            A DistanceWindowGraph
        """

        self._max_distance = max_distance
        self._edge_distance = edge_distance

    @property
    def max_distance(self):
        return self._max_distance

    def add_node(self, node: BaseNode):
        """Adds a node to the graph and removes far nodes"""
        if self._edge_distance is not None and self.get_last_node() is not None:
            # compute distance to last node and do not add the node if it's too close
            d = node.distance_to(self.get_last_node())
            if d < self._edge_distance:
                return False

        # Add node
        out = super().add_node(node)

        # Remove all nodes farther than self._max_distance
        self.remove_nodes_within_radius_range(
            node, min_radius=self._max_distance, max_radius=float("inf"), metric="pose"
        )
        return out


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
        s = BaseNode(timestamp=i, pose_base_in_world=SE3(SO3.identity(), torch.Tensor([i / 10.0, 0, 0])).as_matrix())
        nodes_list.append(s)
        graph.add_node(s)

    # Check graph as list
    assert nodes_list == graph.get_nodes()

    # Check number of nodes
    assert graph.get_num_nodes() == N

    # Check number of edges
    assert graph.get_num_edges() == N - 1

    # Get nodes within radius
    radius = 0.2
    query_node = graph.get_node_with_timestamp(5.0)
    nodes = graph.get_nodes_within_radius_range(query_node, min_radius=0, max_radius=radius)
    for n in nodes:
        d = query_node.distance_to(n)
        assert d <= radius

    # Get nodes within timespan
    nodes = graph.get_nodes_within_timespan(0.0, 3.0, open_interval=True)
    assert len(nodes) == 2

    # Get nodes within timespan
    nodes = graph.get_nodes_within_timespan(0.0, 3.0, open_interval=False)
    assert len(nodes) == 4

    # Remove nodes
    graph.remove_nodes(nodes)
    assert graph.get_num_nodes() == 6

    # Check if we can modify the graph
    for n in graph.get_nodes():
        orig_ts = n.timestamp
        n.timestamp = 2
        assert orig_ts != n.timestamp


def run_temporal_window_graph():
    from wild_visual_navigation.traversability_estimator import BaseNode
    import torch
    from liegroups.torch import SO3, SE3
    import matplotlib.pyplot as plt

    # Create graph
    W = 25
    N = 50
    graph = TemporalWindowGraph(time_window=W)

    nodes_list = []
    for i in range(N):
        t = i
        s = BaseNode(timestamp=t, pose_base_in_world=SE3(SO3.identity(), torch.Tensor([i / 10.0, 0, 0])).as_matrix())
        nodes_list.append(s)
        graph.add_node(s)
        assert graph.get_first_node().timestamp >= t - W


if __name__ == "__main__":
    # run_base_graph()
    run_temporal_window_graph()
