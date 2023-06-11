import csv
import os
from time import time

import numba
import numpy as np
import pandas as pd
from tqdm import tqdm

from .seg_info import seginfo_constructor, gen_vehicle_num
from .graph_embedding import traj_freq
from .utils import load_paths
from microdict import mdict
import geopandas as gpd


class SparseDAM(object):
    def __init__(self, workspace, seg_num, mask_size=-1, lb_csmv=1):
        self.seg_num = seg_num
        self.mask_size = mask_size
        self.lb_csmv = lb_csmv

        data = pd.read_csv(os.path.join(workspace, "csm_all.txt"), sep=" ", header=None, names=['row', 'col', 'value']).to_numpy(dtype=np.int32)

        self.mat_col = np.zeros(seg_num, dtype=object)
        self.mat_row = np.zeros(seg_num, dtype=object)
        cnt = 0
        for i in range(seg_num):
            self.mat_col[i] = mdict.create(dtype="i32:i32")
            self.mat_row[i] = mdict.create(dtype="i32:i32")
        for tmp in data:
            self.mat_col[tmp[1]][tmp[0]] = tmp[2]
            self.mat_row[tmp[0]][tmp[1]] = tmp[2]
            cnt += 1

    def get_csm_value(self, o, d):
        value = self.mat_col[d][o]
        if value is None:
            value = 0
        return value

    def get_rank(self, o, k, d):
        o2k = self.get_csm_value(o, k)
        k2d = self.get_csm_value(k, d)
        res = k2d
        if o2k < res:
            res = o2k
        return res

    def get_col(self, d):
        return self.mat_col[d]

    def get_rank_list(self, o, d):
        src = self.mat_row[o]
        des = self.mat_col[d]

        candidates = []
        for k in des:
            k2d = des[k]
            if k2d >= self.lb_csmv:
                o2k = src[k]
                if o2k is not None:
                    rank = k2d
                    if rank > o2k:
                        rank = o2k
                    if rank >= self.lb_csmv:
                        candidates.append((k, rank))

        candidates.sort(key=lambda elem: elem[0])
        candidates.sort(key=lambda elem: elem[1], reverse=True)
        return candidates[: self.mask_size]


@numba.jit(nopython=False, fastmath=True, forceobj=True)
def get_connection_strength_matrix_sparse(paths, seg_num):
    unigram = np.zeros(seg_num, dtype=np.int32)
    mat_all = [{} for i in range(seg_num)]

    start_time = time()
    for traj in paths:
        length = len(traj)
        for i in range(length):
            u = traj[i]
            unigram[u] += 1
            for j in range(i+1, length):
                if traj[j] in mat_all[u]:
                    mat_all[u][traj[j]] += 1
                else:
                    mat_all[u][traj[j]] = 1
    print("Attn Counting Time:" + '{:.3f}s'.format(time() - start_time))

    data = []
    for i, item in enumerate(mat_all):
        for k, v in item.items():
            data.append([i, k, v])
    print("Non-zero: {}".format(len(data)))
    return unigram, data


def gen_dam(args):
    edges = gpd.read_file(args.edges_shp)
    seg_num = edges.shape[0]
    paths, speeds = load_paths(args.train_file, left=args.left, right=args.right, require_speed=True)
    paths = [item[0] for item in paths]

    speed_info = np.zeros(seg_num, dtype=float)
    for k, v in speeds.items():
        speed_info[k] = np.mean(v)
    print("==> speed size: {}".format(len(speeds)))

    unigram, mat_a = get_connection_strength_matrix_sparse(paths, seg_num)
    with open(os.path.join(args.workspace, "csm_all.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=' ')
        fields_output_file.writerows(mat_a)
    print("Saved csm_all.txt")

    data = seginfo_constructor(edges, speeds=speed_info, freqs=unigram)
    with open(os.path.join(args.workspace, "seg_info.csv"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=' ')
        fields_output_file.writerows(data)
    print("Saved seg_info.csv")

    vehicle_num = gen_vehicle_num(args, seg_num)
    with open(os.path.join(args.workspace, "traffic_num.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=' ')
        fields_output_file.writerows(vehicle_num)
    print("Saved traffic_num.txt")

    data_rows = traj_freq(paths, args.edges_shp)
    with open(os.path.join(args.workspace, "weighted_edges.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=' ')
        fields_output_file.writerows(data_rows)
    print("Saved weighted_edges.txt")


def txt2npy(workspace):
    edges = pd.read_csv(os.path.join(workspace, "seg_info.csv"), sep=" ", header=None, names=['eid', 'src', 'trg', 'len', 'rt', 'geo_src', 'geo_trg', 'azimuth', 'freq', 'travel_time'])
    seg_num = edges.shape[0]

    segs_geo = []
    for i in tqdm(range(seg_num), desc="seg_num"):
        tmp = edges.query("eid == {}".format(i)).iloc[0]
        u = [float(item) for item in tmp['geo_src'].split(",")]
        v = [float(item) for item in tmp['geo_trg'].split(",")]
        segs_geo.append(u + v)
    segs_geo = np.array(segs_geo, dtype=float)
    np.save(os.path.join(workspace, "segs_geo.npy"), segs_geo)

    traffic_data = pd.read_csv(os.path.join(workspace, "traffic_num.txt"), sep=" ", header=None, names=['row', 'col', 'value']).to_numpy(dtype=np.int32)
    time_delta = 3600
    num_1h = int(60 * 60 / time_delta)
    num_1d = 24 * num_1h
    vehicle_num = np.zeros((seg_num, num_1d*2), dtype=np.int32)
    for tmp in tqdm(traffic_data, desc="traffic"):
        vehicle_num[tmp[1], tmp[0]] = tmp[2]
    np.save(os.path.join(workspace, "vehicle_num_{}-{}.npy".format(time_delta, vehicle_num.shape[1])), vehicle_num)

    traffic_popularity = np.array(vehicle_num, dtype=float)
    for t_idx in tqdm(range(traffic_popularity.shape[1]), desc="timeslot num"):
        min_traffic = traffic_popularity[:, t_idx].min()
        max_traffic = traffic_popularity[:, t_idx].max()
        traffic_popularity[:, t_idx] = (traffic_popularity[:, t_idx] - min_traffic) * 2.0 / (max_traffic - min_traffic) - 1
    np.save(os.path.join(workspace, "traffic_popularity.npy"), traffic_popularity)


if __name__ == '__main__':
    pass
