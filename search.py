import math
from collections.abc import Mapping
from heapq import heappush, heappop
from itertools import count
import re


def bfs(matrix, source, target, heightDifference):
    queue = []
    allpath = []
    queue.append([source])
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == target:
            allpath.append(path)
        for adjacent in seekingAdjacentPoints(matrix, heightDifference, node):

            new_path = list(path)
            new_path.append(adjacent)
            m1 = set(new_path)
            if len(m1) == len(new_path):
                queue.append(new_path)
    n = []
    for i in allpath:
        n.append(len(i))
    if len(allpath) == 0:
        return 'FAIL'
    else:
        shortPath = allpath[n.index(min(n))]
        return shortPath


def ucs(matrix, source, target, heightDifference):

    queue = []
    allpath = []

    queue.append([source])
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == target:
            allpath.append(path)
        for adjacent in seekingAdjacentPoints(matrix, heightDifference, node):

            new_path = list(path)
            new_path.append(adjacent)
            m1 = set(new_path)
            if len(m1) == len(new_path):
                queue.append(new_path)
    n = []
    for iPath in allpath:
        cost = 0
        for i in range(1, len(iPath)):
            u = iPath[i].replace("-", " ")
            v = iPath[i - 1].replace("-", " ")
            u = re.split(" ", u)
            v = re.split(" ", v)
            a1 = int(u[0])
            a2 = int(u[1])
            b1 = int(v[0])
            b2 = int(v[1])
            if b2 == a2 or b1 == a1:
                costLu = 10
            else:
                costLu = 10 * math.sqrt(2)
            cost = cost + costLu
        n.append(cost)
    if len(allpath) == 0:
        return 'FAIL'
    else:
        shortPath = allpath[n.index(min(n))]
        return shortPath


def astar_path(matrix, source, target, heightDifference, w=10, heuristic=None, weight='weight'):
    G = Graph()
    H = len(matrix)
    W = len(matrix[0])
    for a in range(H):
        for b in range(W):
            if a - 1 >= 0 and b - 1 >= 0 and a + 1 < H and b + 1 < W:
                if abs(matrix[a][b] - matrix[a - 1][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b - 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a - 1][b]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a - 1][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b + 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a, b - 1),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a, b + 1),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b - 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b + 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
            elif a == 0 and b - 1 >= 0 and a + 1 < H and b + 1 < W:
                if abs(matrix[a][b] - matrix[a][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a, b - 1),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a, b + 1),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b - 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b + 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
            elif a - 1 >= 0 and b == 0 and a + 1 < H and b + 1 < W:
                if abs(matrix[a][b] - matrix[a - 1][b]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a - 1][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b + 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a, b + 1),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b + 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
            elif a - 1 >= 0 and b - 1 >= 0 and a == H - 1 and b + 1 < W:
                if abs(matrix[a][b] - matrix[a - 1][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b - 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a - 1][b]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a - 1][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b + 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a, b - 1),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a, b + 1),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
            elif a - 1 >= 0 and b - 1 >= 0 and a + 1 < H and b == W - 1:
                if abs(matrix[a][b] - matrix[a - 1][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b - 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a - 1][b]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a, b - 1),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b - 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
            elif a == 0 and b == 0 and a < H and b < W:
                if abs(matrix[a][b] - matrix[a][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a, b + 1),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b + 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
            elif a - 1 >= 0 and b - 1 >= 0 and a == H - 1 and b == W - 1:
                if abs(matrix[a][b] - matrix[a - 1][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b - 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a - 1][b]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a, b - 1),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
            elif a - 1 >= 0 and b == 0 and a == H - 1 and b < W - 1:
                if abs(matrix[a][b] - matrix[a - 1][b]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a - 1][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a - 1, b + 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a][b + 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a, b + 1),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
            elif a == 0 and b - 1 >= 0 and a < H - 1 and b == W - 1:
                if abs(matrix[a][b] - matrix[a][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a, b - 1),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b - 1]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b - 1),
                               weight=w * math.sqrt(2) + matrix[a][b] - matrix[a - 1][b - 1])
                if abs(matrix[a][b] - matrix[a + 1][b]) <= heightDifference:
                    G.add_edge("{}-{}".format(a, b), "{}-{}".format(a + 1, b),
                               weight=w + matrix[a][b] - matrix[a - 1][b - 1])
    if source not in G or target not in G:
        return 'FAIL'
    if heuristic is None:
        def heuristic(u, v):
            return 0
            # math.sqrt((b1 - a1) ** 2 + (b2 - a2) ** 2 + (matrix[a1][a2] - matrix[b1][b2]) ** 2)
    push = heappush
    pop = heappop
    c = count()
    queue = [(0, next(c), source, 0, None)]
    enqueued = {}
    explored = {}
    while queue:
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path
        if curnode in explored:
            continue
        explored[curnode] = parent

        for neighbor, w in G[curnode].items():
            if neighbor in explored:
                continue
            ncost = dist + w.get(weight, 1) + heightDifference
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))
    return 'FAIL'


class AtlasView(Mapping):
    __slots__ = ('_atlas',)

    def __getstate__(self):
        return {'_atlas': self._atlas}

    def __setstate__(self, state):
        self._atlas = state['_atlas']

    def __init__(self, d):
        self._atlas = d

    def __len__(self):
        return len(self._atlas)

    def __iter__(self):
        return iter(self._atlas)

    def __getitem__(self, key):
        return self._atlas[key]

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}

    def __str__(self):
        return str(self._atlas)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._atlas)


class AdjacencyView(AtlasView):
    def __getitem__(self, name):
        return AtlasView(self._atlas[name])


class Graph(object):
    node_dict_factory = dict
    node_attr_dict_factory = dict
    adjlist_outer_dict_factory = dict
    adjlist_inner_dict_factory = dict
    edge_attr_dict_factory = dict
    graph_attr_dict_factory = dict

    def __init__(self):
        self.graph_attr_dict_factory = self.graph_attr_dict_factory
        self.node_dict_factory = self.node_dict_factory
        self.node_attr_dict_factory = self.node_attr_dict_factory
        self.adjlist_outer_dict_factory = self.adjlist_outer_dict_factory
        self.adjlist_inner_dict_factory = self.adjlist_inner_dict_factory
        self.edge_attr_dict_factory = self.edge_attr_dict_factory
        self.graph = self.graph_attr_dict_factory()
        self._node = self.node_dict_factory()
        self._adj = self.adjlist_outer_dict_factory()

    @property
    def adj(self):
        return AdjacencyView(self._adj)

    def __contains__(self, n):
        try:
            return n in self._node
        except TypeError:
            return False

    def __getitem__(self, n):
        return self.adj[n]

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        u, v = u_of_edge, v_of_edge
        if u not in self._node:
            self._adj[u] = self.adjlist_inner_dict_factory()
            self._node[u] = self.node_attr_dict_factory()
        if v not in self._node:
            self._adj[v] = self.adjlist_inner_dict_factory()
            self._node[v] = self.node_attr_dict_factory()
        datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
        datadict.update(attr)
        self._adj[u][v] = datadict
        self._adj[v][u] = datadict


def seekingAdjacentPoints(matrix, heightDifference, point):
    point = point.replace("-", " ")
    u = re.split(" ", point)
    a = int(u[0])
    b = int(u[1])
    H = len(matrix)
    W = len(matrix[0])
    kk = []
    if a - 1 >= 0 and b - 1 >= 0 and a + 1 < H and b + 1 < W:
        if abs(matrix[a][b] - matrix[a - 1][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b - 1))
        if abs(matrix[a][b] - matrix[a - 1][b]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b))
        if abs(matrix[a][b] - matrix[a - 1][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b + 1))
        if abs(matrix[a][b] - matrix[a][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a, b - 1))
        if abs(matrix[a][b] - matrix[a][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a, b + 1))
        if abs(matrix[a][b] - matrix[a + 1][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b - 1))
        if abs(matrix[a][b] - matrix[a + 1][b]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b))
        if abs(matrix[a][b] - matrix[a + 1][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b + 1))
    elif a == 0 and b - 1 >= 0 and a + 1 < H and b + 1 < W:
        if abs(matrix[a][b] - matrix[a][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a, b - 1))
        if abs(matrix[a][b] - matrix[a][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a, b + 1))
        if abs(matrix[a][b] - matrix[a + 1][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b - 1))
        if abs(matrix[a][b] - matrix[a + 1][b]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b))
        if abs(matrix[a][b] - matrix[a + 1][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b + 1))
    elif a - 1 >= 0 and b == 0 and a + 1 < H and b + 1 < W:
        if abs(matrix[a][b] - matrix[a - 1][b]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b))
        if abs(matrix[a][b] - matrix[a - 1][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b + 1))
        if abs(matrix[a][b] - matrix[a][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a, b + 1))
        if abs(matrix[a][b] - matrix[a + 1][b]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b))
        if abs(matrix[a][b] - matrix[a + 1][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b + 1))
    elif a - 1 >= 0 and b - 1 >= 0 and a == H - 1 and b + 1 < W:
        if abs(matrix[a][b] - matrix[a - 1][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b - 1))
        if abs(matrix[a][b] - matrix[a - 1][b]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b))
        if abs(matrix[a][b] - matrix[a - 1][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b + 1))
        if abs(matrix[a][b] - matrix[a][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a, b - 1))
        if abs(matrix[a][b] - matrix[a][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a, b + 1))
    elif a - 1 >= 0 and b - 1 >= 0 and a + 1 < H and b == W - 1:
        if abs(matrix[a][b] - matrix[a - 1][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b - 1))
        if abs(matrix[a][b] - matrix[a - 1][b]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b))
        if abs(matrix[a][b] - matrix[a][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a, b - 1))
        if abs(matrix[a][b] - matrix[a + 1][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b - 1))
        if abs(matrix[a][b] - matrix[a + 1][b]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b))
    elif a == 0 and b == 0 and a < H and b < W:
        if abs(matrix[a][b] - matrix[a][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a, b + 1))
        if abs(matrix[a][b] - matrix[a + 1][b]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b))
        if abs(matrix[a][b] - matrix[a + 1][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b + 1))
    elif a - 1 >= 0 and b - 1 >= 0 and a == H - 1 and b == W - 1:
        if abs(matrix[a][b] - matrix[a - 1][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b - 1))
        if abs(matrix[a][b] - matrix[a - 1][b]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b))
        if abs(matrix[a][b] - matrix[a][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a, b - 1))
    elif a - 1 >= 0 and b == 0 and a == H - 1 and b < W - 1:
        if abs(matrix[a][b] - matrix[a - 1][b]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b))
        if abs(matrix[a][b] - matrix[a - 1][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a - 1, b + 1))
        if abs(matrix[a][b] - matrix[a][b + 1]) <= heightDifference:
            kk.append("{}-{}".format(a, b + 1))
    elif a == 0 and b - 1 >= 0 and a < H - 1 and b == W - 1:
        if abs(matrix[a][b] - matrix[a][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a, b - 1))
        if abs(matrix[a][b] - matrix[a + 1][b - 1]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b - 1))
        if abs(matrix[a][b] - matrix[a + 1][b]) <= heightDifference:
            kk.append("{}-{}".format(a + 1, b))
    return kk
