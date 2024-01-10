#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
import multiprocessing
import os
import string
import networkx as nx
import EoN
import numpy as np
import scipy
import igraph as ig
import copy
import core
import time
import algorithms
import layouts
import graphs


# ("Jazz") # 0 Jazz 198 2742
# ("USAir")# 1 USAir 332 2126
# ("Netscience")# 2 Netscience 379 914
# ("NS")# 3 NS 379 914
# ("EEC")# 4 EEC 986 16064
# ("Metabolic")# 5 Metabolic 1039 4741
# ("Email")# 6 Email 1133 5451
# ("Euroroad")# 7 Euroroad 1174 1417
# ("Blogs")# 8 Blogs 1222 16714
# ("PB")# 9 PB 1222 16714
# ("Protein")# 10 protein 2018 2930
# ("Facebook")# 11 Facebook 4039 88234
# ("GrQc")# 12 GrQc 4158 13422
# ("Power")# 13 Power 4941 6594
# ("Powergrid")# 14 powergrid 4941 6594
# ("Router")# 15 Router 5022 6258
# ("PG")# 16 PG 6299 20776
# ("WikiVote")# 17 WikiVote 7066 100736
# ("WV")# 18 WV 7066 100736
# ("Sex")# 19 Sex 15810 38540
# ("Collaboration")# 20 Collaboration 23133 93439
# ("Enron")# 21 Enron 33696 180811
# ("Phonecalls")# 22 Phonecalls 36595 56853
# ("Coauthor")# 23 Coauthor 51079 85771
# ("Internet")# 24 Internet 192244 609066
# ("WWW")# 25 www 325729 1117563
# ("Citation")# 26 citation 449673 4685576
# ("Actor")# 27 actor 702388 29397908
# 0 Internet (AS level) (1) 10515 21455
# 1 p2p-Gnutella04 10876 39994
# 2 Coauthor (2) 12008 118521
# 3 ca-AstroPh 18771 198050
# 4 CA-CondMat 23133 93497
# 5 ego-twitter 23370 32831
# 6 Google+ (NIPS) 23628 39194
# 7 Internet (AS level) (2) 26475 53381
# 8 Email-Enron 36692 183831
# 9 brightkite_edges 58228 214078
# 10 Internet (router level) 192244 609066
# 11 WWW (2) 325729 1117563
# 12 amazon 334863 925872

NET000 = [
    "Jazz",
    "USAir",
    "Netscience",
    "EEC",
    "Metabolic",
    "Email",
    "Blogs",
    "Facebook",
    "GrQc",
    "Powergrid",
    "Router",
    "PG",
    "WikiVote",
    "Internet(AS)",
    "p2p-Gnutella04",
    "ca-AstroPh",
    # "Sex",
    # "Collaboration",
    # "Enron",
    # "Phonecalls",
    # "Coauthor",
    # "Internet",
    # "WWW",
    # "Citation",
    # "Actor",
]
NET001 = ["Example"]
NET002 = ["Instance"]
NET009 = [
    "Jazz",
    "USAir",
    "NS",
    "EEC",
    "Metabolic",
    "Email",
    "Euroroad",
    "PB",
    "Facebook",
]
NET012 = [
    "EEC",
    "Metabolic",
    "Email",
    "Euroroad",
    "PB",
    "Facebook",
    "GrQc",
    "Powergrid",
    "Router",
    "PG",
    "WV",
    "Sex",
]
NET016 = [
    "Jazz",
    "USAir",
    "Netscience",
    "EEC",
    "Metabolic",
    "Email",
    "Euroroad",
    "Blogs",
    "Facebook",
    "GrQc",
    "Powergrid",
    "Router",
    "PG",
    "WikiVote",
    "Internet(AS)",
    "Sex",
]
NET020 = [
    "Jazz",
    "USAir",
    "Netscience",
    "EEC",
    "Metabolic",
    "Email",
    "Euroroad",
    "Blogs",
    "Facebook",
    "GrQc",
    "Powergrid",
    "Router",
    "PG",
    "WikiVote",
    "Internet(AS)",
    "p2p-Gnutella04",
    "ca-AstroPh",
    "Sex",
    "Collaboration",
    "Enron",
]

# 获取当前脚本所在目录
current_directory = os.path.dirname(os.path.abspath(__file__))



def load_networks(data=NET002):
    networks = []
    for index, name in enumerate(data):
        TIME = time.time()
        G = nx.read_edgelist(
            os.path.join(current_directory, "data", name + ".txt"),
            create_using=nx.Graph(),
            nodetype=np.int64,
        )
        G.N = G.number_of_nodes()
        G.M = G.number_of_edges()
        G.name = name
        networks.append(G)
        print(index, G.name, time.time() - TIME)
    networks.sort(key=lambda x: x.N)
    for i, G in enumerate(networks):
        print(i, G.name, G.N, G.M)
        G.number = string.ascii_letters[i]
        G.index = i
    return networks


def Core(G):
    return core.Core(G)


# Load real network data to generate the network
def load_graph_data(name):
    G = nx.read_edgelist("./network_edgelist/" + name + ".edgelist", nodetype=np.int64)
    N = nx.number_of_nodes(G)
    mapping = dict(zip(G, range(N)))
    G = nx.relabel_nodes(G, mapping)
    return G


def get_average_degree(N, M):
    return 2 * M / N


def get_H_index(G):
    # 计算每个节点的度数
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

    # 计算度异质性指数
    return np.sum(
        (degree_sequence[i] - np.mean(degree_sequence)) ** 2
        / (np.std(degree_sequence) ** 2)
        for i in range(len(degree_sequence))
    )


# epidemic threshold: beta_c
def get_beta_c(G, N):
    k = sum([G.degree(i) for i in G.nodes()]) / N
    square_k = sum([G.degree(i) ** 2 for i in G.nodes()]) / N

    return k / (square_k - k)


# DC^{+}
def DC_plus(G):
    av_nei_deg = nx.average_neighbor_degree(G)
    DC_AND = {i: G.degree(i) * av_nei_deg[i] for i in G.nodes()}

    return DC_AND


# Calculate the shortest path distance matrix of the network
def get_distance_matrix(G, N):
    g = ig.Graph.from_networkx(G)
    DM = np.array(g.shortest_paths()).reshape(N, N)

    return DM


# references: Li H, Shang Q, Deng Y. A generalized gravity model for influential spreaders identification in complex networks[J].
# Chaos, Solitons & Fractals, 2021, 143: 110456.
def cal_SP(G):
    SP = {}
    for i in G.nodes():
        ci = nx.clustering(G, i)
        SP[i] = np.exp(-2.0 * ci) * G.degree(i)
    return SP


def EHCC(G):
    return EHCC_main(G, 0.5)


# gravity model
def DCGM(G):
    R, nodes, DC = 2, list(G.nodes()), dict(nx.degree(G))
    g = ig.Graph.from_networkx(G)
    DCGM = {}

    for i in nodes:
        s1 = 0
        ball_i_R = set(g.neighborhood(i, order=R))
        ball_i_R.remove(i)
        for j in ball_i_R:
            dij = g.distances(i, j)[0][0]
            if dij > 0:
                s1 += (DC[i] * DC[j]) / (dij**2)
        DCGM[i] = s1

    return DCGM


def GGC(G):
    R, nodes, SP = 2, list(G.nodes()), cal_SP(G)
    g = ig.Graph.from_networkx(G)
    GGC = {}

    for i in nodes:
        s2 = 0
        ball_i_R = set(g.neighborhood(i, order=R))
        ball_i_R.remove(i)
        for j in ball_i_R:
            dij = g.distances(i, j)[0][0]
            if dij > 0:
                s2 += (SP[i] * SP[j]) / (dij**2)
        GGC[i] = s2

    return GGC


def DCGM_plus(G):
    R, nodes, DC_AND = 2, list(G.nodes()), DC_plus(G)
    g = ig.Graph.from_networkx(G)
    DCGM1 = {}

    for i in nodes:
        s3 = 0
        ball_i_R = set(g.neighborhood(i, order=R))
        ball_i_R.remove(i)
        for j in ball_i_R:
            dij = g.distances(i, j)[0][0]
            if dij > 0:
                s3 += (DC_AND[i] * DC_AND[j]) / (dij**2)
        DCGM1[i] = s3

    return DCGM1


def DCGM_plus_plus(G):
    R, nodes, DC, DC_AND = 2, list(G.nodes()), dict(nx.degree(G)), DC_plus(G)
    g = ig.Graph.from_networkx(G)
    DCGM2 = {}

    for i in nodes:
        s1 = 0
        ball_i_R = set(g.neighborhood(i, order=R))
        ball_i_R.remove(i)
        for j in ball_i_R:
            dij = g.distances(i, j)[0][0]
            if dij > 0:
                s1 += (DC[i] * DC[j]) / (dij**2)
        DCGM2[i] = DC_AND[i] * s1

    return DCGM2


def GM_model(R, nodes, DM, DC, SP, DC_AND):
    DCGM = {}
    GGC = {}
    DCGM1 = {}
    DCGM2 = {}

    for i in nodes:
        s1 = 0
        s2 = 0
        s3 = 0
        index_j = np.argwhere(DM[i] <= R).flatten()
        for j in index_j:
            dij = DM[i, j]
            if dij > 0:
                s1 += (DC[i] * DC[j]) / (dij**2)
                s2 += (SP[i] * SP[j]) / (dij**2)
                s3 += (DC_AND[i] * DC_AND[j]) / (dij**2)
        DCGM[i] = s1
        GGC[i] = s2
        DCGM1[i] = s3
        DCGM2[i] = DC_AND[i] * s1

    return DCGM, GGC, DCGM1, DCGM2


def GM_model2(G, R, nodes, DC, SP, DC_AND):
    g = ig.Graph.from_networkx(G)
    DCGM = {}
    GGC = {}
    DCGM1 = {}
    DCGM2 = {}

    for i in nodes:
        s1 = 0
        s2 = 0
        s3 = 0
        ball_i_R = set(g.neighborhood(i, order=R))
        ball_i_R.remove(i)
        for j in ball_i_R:
            dij = g.distances(i, j)[0][0]
            if dij > 0:
                s1 += (DC[i] * DC[j]) / (dij**2)
                s2 += (SP[i] * SP[j]) / (dij**2)
                s3 += (DC_AND[i] * DC_AND[j]) / (dij**2)
        DCGM[i] = s1
        GGC[i] = s2
        DCGM1[i] = s3
        DCGM2[i] = DC_AND[i] * s1

    return DCGM, GGC, DCGM1, DCGM2


def GSM(G):
    KS = nx.core_number(G)
    DM = get_distance_matrix(G, G.N)
    nodes = list(G.nodes())
    GSM = {}
    for i in nodes:
        s = 0
        for j in nodes:
            dij = DM[i, j]
            if dij > 0:
                s += KS[j] / dij
        GSM[i] = math.exp(KS[i] / G.N) * s

    return GSM


def IGSM(G):
    DC = dict(nx.degree(G))
    ave_DC = get_average_degree(G.N, G.M)
    DM = get_distance_matrix(G, G.N)
    nodes = list(G.nodes())
    IGSM = {}
    for i in nodes:
        s = 0
        for j in nodes:
            dij = DM[i, j]
            if dij > 0:
                s += DC[j] / dij ** math.ceil(math.log2(ave_DC))
        IGSM[i] = math.exp(DC[i] / G.N) * s

    return IGSM


def HGSM(G):
    KS = nx.core_number(G)
    DC = dict(nx.degree(G))
    DM = get_distance_matrix(G, G.N)
    nodes = list(G.nodes())
    HGSM = {}
    for i in nodes:
        avs = 0
        n = 0
        for j in nodes:
            dij = DM[i, j]
            if dij > 0:
                n += 1
                avs += math.exp(KS[j] * DC[j] / G.N)
        if n > 0:
            avs /= n
        s = 0
        for j in nodes:
            dij = DM[i, j]
            if dij > 0 and dij ** math.ceil(math.log2(avs)) > 0:
                s += math.exp(KS[j] * DC[j] / G.N) / dij ** math.ceil(math.log2(avs))
        HGSM[i] = math.exp(KS[i] * DC[i] / G.N) * s

    return HGSM


def get_SIR_ranking(G, beta, gamma, tmax, report_times, iterations):
    SR = {}
    for i in G.nodes():
        obs_R = 0 * report_times
        for counter in range(iterations):
            t, S, I, R = EoN.fast_SIR(G, beta, gamma, initial_infecteds=i, tmax=tmax)
            obs_R += EoN.subsample(report_times, t, R)

        SR[i] = obs_R[-1] / iterations

    return SR


def cal_Kendall_tau_coefficient(X, Y):
    tau, p_value = scipy.stats.kendalltau(X, Y)
    return tau


def cal_spearman_r_coefficient(X, Y):
    a = scipy.stats.spearmanr(X, Y)
    return a[0]


def cal_SIR(G):
    nums = 11
    gamma = 1.0  # recovery rate
    tmin, tmax = 0.0, 50.0
    iterations = 1000
    report_times = np.linspace(tmin, tmax, 21)

    N, M = len(G.nodes()), len(G.edges())

    # epidemic threshold: beta_c
    beta_c = get_beta_c(G, N)

    beta_list = np.linspace(0.5, 1.5, nums) * beta_c  # infection probability $\beta$

    SR = np.zeros((N, nums + 1))  # standard ranking
    for j, beta in enumerate(beta_list):
        for i in G.nodes():
            obs_R = 0 * report_times
            for counter in range(iterations):
                t, S, I, R = EoN.fast_SIR(
                    G, beta, gamma, initial_infecteds=i, tmax=tmax
                )
                obs_R += EoN.subsample(report_times, t, R)

            SR[i, j + 1] = obs_R[-1] / iterations
    SR[:, 0] = np.array(
        list(G.nodes())
    )  # The first column holds the node labels. Note that node labels range from 0 to N-1.
    return SR


def cal_Monotonocity(x_list, y_list):
    n = len(x_list)
    n_r = 0
    for index, item in enumerate(x_list):
        if int(x_list[index]) == int(y_list[index]):
            n_r = n_r + 1

    return (1 - ((n_r * (n_r - 1)) / (n * (n - 1)))) ** 2


def arg_min(x):  # argmin function for dict type
    """
    x: a dict
    """
    y = min([x[key] for key in x.keys()])  # the minimum value
    z = []  # contains keys with the minimum value
    for key in x.keys():
        if x[key] == y:
            z.append(key)
    return z


def ex_deg(g, delta):  # extended degree
    dict_exdeg = {}  # contains node extended degree
    for node in g.nodes():
        exdeg = delta * g.degree[node]
        for neighbor in g.neighbors(node):
            exdeg += (1 - delta) * g.degree[neighbor]
        dict_exdeg[node] = exdeg  # calculate node extended degree
    return dict_exdeg


def E_shell_decomp(g, delta):  # E-shell hierarchy decomposition
    pos = {}  # contain position index of nodes
    pos_index = 0  # position index
    while g:
        pos_index += 1
        dict_exdeg = ex_deg(g, delta)  # calculate extended degree for current network
        min_nodes = arg_min(dict_exdeg)  # find the nodes with minimum extended degree
        for i in min_nodes:
            pos[i] = pos_index  # assign position index to min nodes
        g.remove_nodes_from(min_nodes)  # delete min nodes from current network
    return pos


def EHCC_main(g, delta):  # main program of EHCC
    """
    g: input a network
    delta: a weight parameter in [0,1]
    """
    g_1 = copy.deepcopy(g)  # copy the network
    extended_degree = ex_deg(g, delta)  # calculate extended degree
    pos = E_shell_decomp(g_1, delta)  # calculate position index
    max_exdeg = max(
        [extended_degree[node] for node in g.nodes()]
    )  # maximal extended degree
    max_pos = max([pos[node] for node in g.nodes()])  # maximal position index
    hcc = {}  # contain hcc value of nodes
    for node in g.nodes():
        hcc[node] = extended_degree[node] / max_exdeg + pos[node] / max_pos
    ehcc = {}  # contain ehcc value of nodes
    for node in g.nodes():
        temp = hcc[node]
        for neighbor in g.neighbors(node):
            temp += hcc[neighbor]
        ehcc[node] = temp
    return ehcc


def io(G):
    csv = os.path.join(
        current_directory,
        "results",
        "algorithm_" + G.name + "_SIR.csv",
    )
    if not os.path.exists(csv):
        return
    
    nums = 11
    gamma = 1.0  # recovery rate
    tmin, tmax = 0.0, 50.0
    iterations = 1000
    report_times = np.linspace(tmin, tmax, 21)
    beta_c = get_beta_c(G, G.N)
    beta_list = np.linspace(0.5, 1.5, nums) * beta_c  # infection probability $\beta$
    
    SR = np.loadtxt(
        csv,
        delimiter=",",
        dtype=np.float64,
    )
    for i, beta in enumerate(beta_list):
        csv = os.path.join(
            current_directory,
            "results",
            "algorithm_" + G.name + "_SIR-" + str(round(beta / beta_c, 1)) + ".csv",
        )
        os.makedirs(os.path.dirname(csv), exist_ok=True)
        if not os.path.exists(csv):
            X = dict(zip(SR[:, 0], SR[:, i + 1]))
            Y = dict(sorted(X.items(), key=lambda x: x[0]))
            R = np.empty((G.N, 2))
            R[:, 0] = np.array(list(Y.keys()))
            R[:, 1] = np.array(list(Y.values()))
            np.savetxt(
                csv,
                R,
                delimiter=",",
                fmt="%f",
            )


def f(G):
    algorithms.f(G)
    layouts.f(G)
    graphs.f(G)
    # circles.f(G)
    # magnifiers.f(G)


if __name__ == "__main__":
    data = load_networks()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
        p.map(f, data)
