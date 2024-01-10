import json
import multiprocessing
import os
import numpy as np
import utils
import networkx as nx


def f(G):
    path = os.path.join(utils.current_directory, "results")
    os.makedirs(path, exist_ok=True)
    csv = os.path.join(path, "layout_" + G.name + "_spring.csv")
    if os.path.exists(csv):
        return
    data = nx.spring_layout(G, iterations=1000)
    R = np.empty((G.N, 3))
    R[:, 0] = np.array(list(data.keys()))
    for i, key in enumerate(data.keys()):
        R[i, 1:] = data[key]
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