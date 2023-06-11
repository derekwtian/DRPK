import argparse
import pickle

import numpy as np
import networkx as nx
import os
from time import time

import pandas as pd
from tqdm import tqdm

from .ge import Node2Vec

from .utils import load_edges
import geopandas as gpd


def get_road_graph(workspace):
    edges = pd.read_csv(os.path.join(workspace, "seg_info.csv"), sep=" ", header=None, names=['eid', 'source', 'target', 'length', 'rt', 'geo_src', 'geo_trg', 'azimuth', 'freq', 'travel_time'])
    print("Number of Segments: {}".format(edges.shape[0]))

    G = nx.DiGraph(nodetype=int)

    eids = edges['eid'].tolist()
    lengths = edges['length'].tolist()
    times = edges['travel_time'].tolist()
    targets = edges['target'].tolist()

    for i in tqdm(range(len(eids)), desc="road_graph"):
        u = eids[i]
        tar = targets[i]
        out_edges = edges.query('source == ' + str(tar))
        v_set = out_edges['eid'].tolist()
        out_length = out_edges['length'].tolist()
        out_time = out_edges['travel_time'].tolist()
        for k in range(len(v_set)):
            G.add_edge(u, v_set[k], length=round(out_length[k], 3), time=round(out_time[k], 3))
        # add length attribute for node (segment)
        lens = round(lengths[i], 3)
        ts = round(times[i], 3)
        if G.has_node(u):
            G.nodes[u]['length'] = lens
            G.nodes[u]['time'] = ts
        else:
            G.add_node(u, length=lens, time=ts)

    pickle.dump(G, open(os.path.join(workspace, "road_graph_wtime"), "wb"))


def get_edge_list(edges):
    eids = edges['eid'].tolist()
    targets = edges['target'].tolist()
    edge_list = {}
    for i in range(len(eids)):
        u = eids[i]
        tar = targets[i]
        v_set = edges.query('source == {}'.format(tar))['eid'].tolist()
        for v in v_set:
            key = str(u) + " " + str(v)
            edge_list[key] = 1
    return edge_list


def traj_freq(paths, edges_shp):
    segs = load_edges(edges_shp)
    data = get_edge_list(segs)

    # use trajectories or not
    for path in tqdm(paths, desc="traj num"):
        for i in range(len(path)-1):
            key = str(path[i]) + " " + str(path[i+1])
            data[key] += 1

    data_rows = []
    for key, value in data.items():
        tmp = key.split(" ")
        data_rows.append([int(tmp[0]), int(tmp[1]), value])
    return data_rows


def gen_node2vec_emb(args):
    edges = gpd.read_file(args.edges_shp)
    seg_num = edges.shape[0]

    edges_weight_file = "weighted_edges.txt"
    G = nx.read_edgelist(os.path.join(args.workspace, edges_weight_file), create_using=nx.DiGraph(), nodetype=int, data=[('weight', int)])
    print("Segment Nodes: {}, Edges: {}".format(len(G.nodes), len(G.edges)))

    model = Node2Vec(G, walk_length=args.walk_len, num_walks=args.num_walks, p=args.p, q=args.q, workers=32, use_rejection_sampling=0)

    start_time = time()
    model.train(embed_size=args.emb_dim, window_size=args.window, epoch=args.epoch)
    print("Training Time:" + '{:.3f}s'.format(time() - start_time))

    embeddings = model.get_embeddings()  # dict
    print("Expected: ({}, {})".format(seg_num, args.emb_dim))
    print("Obtained: ({}, {})".format(len(embeddings), len(embeddings[0])))

    weights = []
    for i in range(seg_num):
        if i in embeddings:
            weights.append(embeddings[i])
        else:
            print("Both out_degree and in_degree for Node {} are 0".format(i))
            weights.append(np.random.normal(0, 1, size=args.emb_dim))

    seg_embedding = np.array(weights, dtype=np.float32)
    print(seg_embedding.shape)
    np.save(os.path.join(args.workspace, "segs_embedding_{}.npy".format(seg_embedding.shape[1])), seg_embedding)
    print("Node2Vec Embedding Saved Successful")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default="../data/chengdu_NMLR")
    parser.add_argument('--traj_file', type=str, default="traj_20161010")
    parser.add_argument('--edges_shp', type=str, default="/Users/tianwei/dataset/preprocessed_data/chengdu_data/map/edges.shp")

    parser.add_argument('-walk_len', type=int, default=30)
    parser.add_argument('-num_walks', type=int, default=25)
    parser.add_argument('-p', type=float, default=2)
    parser.add_argument('-q', type=float, default=0.25)
    parser.add_argument('-emb_dim', type=int, default=64)
    parser.add_argument('-window', type=int, default=5)
    parser.add_argument('-epoch', type=int, default=200)
    args = parser.parse_args()
    print(args)

    gen_node2vec_emb(args)
