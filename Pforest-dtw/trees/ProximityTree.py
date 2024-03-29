import sys
from random import random
from trees import Node
from core.TreeStatCollector import TreeStatCollector
from trees import DistanceMeasure as dm

"""
A tree has:
- id
- root
- 
"""


class ProximityTree:

    def __init__(self, id, forest, depth=0):
        self.random = random()
        self.node_counter = 0
        self.tree_depth = depth
        self.time_best_splits = 0
        self.root = None
        self.id = id
        if forest is not None:
            self.proximity_forest_id = forest.get_forest_ID()
            self.stats = TreeStatCollector(id, self.proximity_forest_id)
            self.distance_measure = forest.distance_measure
            self.distance_kwargs = forest.distance_kwargs
        else:
            self.distance_measure = dm.DistanceMeasure.find_closest_nodes
            self.distance_kwargs = dict()

    def get_root_node(self):
        return self.root

    def train(self, data):
        self.node_counter = self.node_counter + 1
        self.root = Node.Node(parent=None, label=None, node_id=self.node_counter, depth=self.tree_depth, tree=self, distance_measure=self.distance_measure, distance_kwargs=self.distance_kwargs)
        self.root.train(data)

    def predict(self, query):
        node = self.root
        if node is None:
            return -1
        while not node.is_leaf:
            posicion = node.splitter.find_closest_branch_(query, distance_measure=self.distance_measure, distance_kwargs=self.distance_kwargs)
            if posicion == -1:
                node.is_leaf = True
                continue
            node = node.children[posicion]
        return node.label

    def get_treestat_collection(self):
        self.stats.collate_results(self)
        return self.stats

    def get_num_nodes(self):
        return self._get_num_nodes(self.root)

    def _get_num_nodes(self, node):
        count = 0
        if node.children is None:
            return 1
        for i in range(0, len(node.children)):
            count = count + self._get_num_nodes(node.children[i])
        return count + 1

    def get_num_leaves(self):
        return self._get_num_leaves(self.root)

    def _get_num_leaves(self, n):
        count = 0
        if n.children is None or n.children == 0:
            return 1
        for i in range(0, len(n.children)):
            count = count + self._get_num_leaves(n.children[i])
        return count

    def get_num_internal_node(self):
        return self._get_num_internal_node(self.root)

    def _get_num_internal_node(self, n):
        count = 0
        if n.children is None:
            return 0
        for i in range(0, len(n.children)):
            count = count + self._get_num_internal_node(n.childen[i])

        return count + 1

    def get_height(self):
        return self._get_height(self.root)

    def _get_height(self, n):
        max_depth = 0
        if n.children is None or n.children == 0:
            return 0
        for i in range(0, len(n.children)):
            max_depth = max(max_depth, self._get_height(n.children[i]))
        return max_depth + 1

    def get_min_depth(self, node):
        max_depth = 0
        if node.children is not None:
            return 0
        for i in range(0, len(node.children)):
            max_depth = min(max_depth, self._get_height(node.children[i]))
        return max_depth + 1

