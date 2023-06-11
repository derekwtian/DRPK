import argparse
import csv
import os
from ast import literal_eval

import pandas as pd
from tqdm import tqdm
import geopandas as gpd

from preprocess.seg_info import get_road_type, calc_azimuth


def seginfo_out(workspace, edges_shp):
    edges = gpd.read_file(edges_shp)
    data = []
    for i in range(edges.shape[0]):
        tmp = edges.iloc[i]
        start = tmp['geometry'].coords[0]
        end = tmp['geometry'].coords[-1]
        azimuth = calc_azimuth([float(start[0]), float(start[1]), float(end[0]), float(end[1])])
        eid = tmp['fid']
        length = tmp['length']
        rt = get_road_type(tmp['highway'])
        data.append([eid,
                     tmp['u'],
                     tmp['v'],
                     round(length, 3),
                     rt,
                     "{},{}".format(start[0], start[1]),
                     "{},{}".format(end[0], end[1]),
                     azimuth])

    print(len(data))
    with open(os.path.join(workspace, "edges.csv"), "w") as fp:
        fields_output_file = csv.writer(fp, delimiter=' ')
        fields_output_file.writerows(data)


def reformat_trajs(workspace, filename):
    trajfile = os.path.join(workspace, filename)
    trajs = pd.read_csv(trajfile, sep=",", header=None, names=['oid', 'tid', 'offsets', 'path'])
    print("Trajectories Number: {}".format(trajs.shape[0]))
    data = []
    for i in tqdm(range(trajs.shape[0]), desc="trajectory"):
        tmp = trajs.iloc[i]
        traj = literal_eval(tmp['path'])
        offset = literal_eval(tmp['offsets'])
        offset_o = offset[0]
        offset_d = offset[1]
        seg, timestamp, speed, travel_time = zip(*traj)
        row = [
            tmp['oid'],
            tmp['tid'],
            ",".join([str(item) for item in offset_o]),
            ",".join([str(item) for item in offset_d]),
            ",".join([str(item) for item in seg]),
            ",".join([str(item) for item in timestamp]),
            ",".join([str(item) for item in speed]),
            ",".join([str(item) for item in travel_time]),
        ]
        data.append(row)
    print("Paths Number: {}".format(len(data)))
    with open(os.path.join(workspace, "serialized_" + filename), "w") as fp:
        fields_output_file = csv.writer(fp, delimiter=' ')
        fields_output_file.writerows(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default="data/ptl_100")
    parser.add_argument('-in_file', type=str, default="traj_valid")
    parser.add_argument('--edges_shp', type=str, default="/Users/tianwei/dataset/preprocessed_data/chengdu_data/map/edges.shp")
    parser.add_argument('-mode', type=int, default=2)
    args = parser.parse_args()
    print(args)

    if args.mode == 0:
        reformat_trajs(args.workspace, args.in_file)
    elif args.mode == 1:
        seginfo_out(args.workspace, args.edges_shp)
    print("done")
