#!/usr/bin/python3
# -*- coding: utf-8 -*-
import copy
import math
import multiprocessing
import os
import random
import numpy as np
import networkx as nx
import utils
import algorithms


def Core(G):
    C = nx.average_clustering(G)
    x = nx.core_number(G)
    
    R = (1 - C) * 3
    index = 0
    while index < R:
        obj = copy.deepcopy(x)
        for node in G.nodes():
            neighbors_value = 0
            for neighbor in G.neighbors(node):
                neighbors_value += obj[neighbor]
            x[node] += neighbors_value * (1 - C) / 2
        index += 1
    
    R = (1 - C) * 3
    index = 0
    while index < R:
        obj = copy.deepcopy(x)
        for node in G.nodes():
            neighbors_value = 0
            for neighbor in G.neighbors(node):
                if random.random() < 0.5:
                    neighbors_value += 1 - 1 / (1 + obj[neighbor])
            
            if random.random() < 0.5:
                neighbors_value += 1 - 1 / (1 + neighbors_value)
            x[node] += neighbors_value / G.degree(node)
            
        index += 1

    return x


def f(G):
    path = os.path.join(utils.current_directory, "tables")
    os.makedirs(path, exist_ok=True)
    key = algorithms.gallery()[-1]
    csv = os.path.join(
        utils.current_directory, "results", "algorithm_" + G.name + "_" + key + ".csv"
    )
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    R = np.empty((G.N, 2))
    X = utils.Core(G)
    Y = dict(sorted(X.items(), key=lambda x: x[0]))
    R[:, 0] = np.array(list(Y.keys()))
    R[:, 1] = np.array(list(Y.values()))
    np.savetxt(
        csv,
        R,
        delimiter=",",
        fmt="%f",
    )


if __name__ == "__main__":
    data = utils.load_networks()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
        p.map(f, data)
