import multiprocessing
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import utils

suffixes = ["origin"]

def f(G):
    alpha = 1 / 2 ** (len(str(G.N)) * 1)
    options = {
        "node_size": 1,
        "node_color": "black",
        "edgecolors": "black",
        "alpha": alpha,
    }
    path = os.path.join(utils.current_directory, "results")
    os.makedirs(path, exist_ok=True)
    png = os.path.join(path, "figure_graph_" + G.name + "_" + suffixes[0] + ".png")
    eps = os.path.join(path, "figure_graph_" + G.name + "_" + suffixes[0] + ".eps")
    if os.path.exists(png) and os.path.exists(eps):
        return
    csv = os.path.join(utils.current_directory, "results", "layout_" + G.name + "_spring.csv")
    pos = {}
    R = np.loadtxt(csv, delimiter=",", dtype=np.float64)
    for i in R[:, 0]:
        pos[np.int64(i)] = R[int(i), 1:]

    nx.draw_networkx(G, pos, True, False, **options)
    plt.axis("off")
    plt.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(eps, format="eps", bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    data = utils.load_networks()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
        p.map(f, data)