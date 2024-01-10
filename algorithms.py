#!/usr/bin/python3
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import utils
import EoN
import os
import multiprocessing


MED000 = ["DC", "GGC", "IGSM", "HGSM", "EHCC", "DC+", "DCGM+", "KSDPA"]


def gallery(G=None, medthods=MED000):
    if G == None:
        return medthods

    ty = {}

    ty["DC"] = algorithm(G, "DC", lambda G: nx.degree_centrality(G))
    ty["KS"] = algorithm(G, "KS", lambda G: nx.core_number(G))
    ty["BC"] = algorithm(G, "BC", lambda G: nx.betweenness_centrality(G))
    ty["CC"] = algorithm(G, "CC", lambda G: nx.closeness_centrality(G))
    ty["EC"] = algorithm(G, "EC", lambda G: nx.eigenvector_centrality(G))
    ty["DC+"] = algorithm(G, "DC+", lambda G: utils.DC_plus(G))
    ty["EHCC"] = algorithm(G, "EHCC", lambda G: utils.EHCC(G))
    ty["DCGM"] = algorithm(G, "DCGM", lambda G: utils.DCGM(G))
    ty["GGC"] = algorithm(G, "GGC", lambda G: utils.GGC(G))
    ty["DCGM+"] = algorithm(G, "DCGM+", lambda G: utils.DCGM_plus(G))
    ty["DCGM++"] = algorithm(G, "DCGM++", lambda G: utils.DCGM_plus_plus(G))
    ty["GSM"] = algorithm(G, "GSM", lambda G: utils.GSM(G))
    ty["IGSM"] = algorithm(G, "IGSM", lambda G: utils.IGSM(G))
    ty["HGSM"] = algorithm(G, "HGSM", lambda G: utils.HGSM(G))
    ty["KSDPA"] = algorithm(G, "KSDPA", lambda G: utils.Core(G))

    results = {}
    for i in medthods:
        results[i] = ty[i]
    return results


def algorithm(G, key, fun):
    csv = os.path.join(
        utils.current_directory, "results", "algorithm_" + G.name + "_" + key + ".csv"
    )
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    if not os.path.exists(csv):
        R = np.empty((G.N, 2))
        X = fun(G)
        Y = dict(sorted(X.items(), key=lambda x: x[0]))
        R[:, 0] = np.array(list(Y.keys()))
        R[:, 1] = np.array(list(Y.values()))
        np.savetxt(
            csv,
            R,
            delimiter=",",
            fmt="%f",
        )
    R = np.loadtxt(csv, delimiter=",", dtype=np.float64)
    return dict(zip(R[:, 0], R[:, 1]))


def f(G):
    gallery(G)
    
    nums = 11
    gamma = 1.0  # recovery rate
    tmin, tmax = 0.0, 50.0
    iterations = 1000
    report_times = np.linspace(tmin, tmax, 21)
    # epidemic threshold: beta_c
    beta_c = utils.get_beta_c(G, G.N)
    beta_list = np.linspace(0.5, 1.5, nums) * beta_c  # infection probability $\beta$

    for beta in beta_list:
        csv = os.path.join(
            utils.current_directory,
            "results",
            "algorithm_" + G.name + "_SIR-" + str(round(beta / beta_c, 1)) + ".csv",
        )
        os.makedirs(os.path.dirname(csv), exist_ok=True)
        if os.path.exists(csv):
            continue
        X = {}
        for i in G.nodes():
            obs_R = 0 * report_times
            for counter in range(iterations):
                t, S, I, SR = EoN.fast_SIR(
                    G, beta, gamma, initial_infecteds=i, tmax=tmax
                )
                obs_R += EoN.subsample(report_times, t, SR)
            X[i] = obs_R[-1] / iterations
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


if __name__ == "__main__":
    data = utils.load_networks()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
        p.map(f, data)
