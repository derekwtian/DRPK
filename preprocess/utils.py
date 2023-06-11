import pickle
from ast import literal_eval
from collections import OrderedDict

import numpy as np
import pandas as pd
import geopandas as gpd


def load_edges(edges_shp):
    edges = gpd.read_file(edges_shp)
    eid = edges['fid'].tolist()
    u = edges['u'].tolist()
    v = edges['v'].tolist()
    length = edges['length'].tolist()
    data = []
    for i in range(len(eid)):
        data.append([eid[i], u[i], v[i], length[i]])
    df = pd.DataFrame(data, columns=['eid', 'source', 'target', 'length'])
    print("Number of Segments: {}, {}".format(df.shape[0], edges.shape[0]))
    return df


def load_road_graph(graph_pkl):
    G = pickle.load(open(graph_pkl, "rb"))
    print("Segment Nodes: {}, Edges: {}".format(len(G.nodes), len(G.edges)))
    return G


def load_paths(trajfile, left=3, right=81, require_speed=False):
    trajs = pd.read_csv(trajfile, sep=",", header=None, names=['oid', 'tid', 'offsets', 'path'])
    print("Trajectories Number: {}".format(trajs.shape[0]))
    paths = []
    offsets = []
    speeds = {}
    for i in range(trajs.shape[0]):
        tmp = trajs.iloc[i]
        traj = literal_eval(tmp['path'])
        if left <= len(traj) < right:
            path = []
            for seg in traj:
                segid = seg[0]
                path.append(segid)
                if require_speed:
                    seg_speed = seg[2]
                    if seg_speed >= 0.1:
                        if segid not in speeds:
                            speeds[segid] = []
                        speeds[segid].append(seg_speed)
            paths.append(path)
            offsets.append(literal_eval(tmp['offsets']))
    print("Paths Number: {}".format(len(paths)))
    return list(zip(paths, offsets)), speeds


def load_paths2(trajfile, left=3, right=81):
    trajs = pd.read_csv(trajfile, sep=",", header=None, names=['oid', 'tid', 'offsets', 'path'])
    print("Trajectories Number: {}".format(trajs.shape[0]))
    res = []
    for i in range(trajs.shape[0]):
        tmp = trajs.iloc[i]
        traj = literal_eval(tmp['path'])
        if left <= len(traj) < right:
            res.append(traj)
    print("Paths Number: {}".format(len(res)))
    return res


def calc_cos_value(vec1, vec2):
    a = vec1 * vec1
    b = vec2 * vec2
    c = vec1 * vec2
    denom = np.sqrt(a[0] + a[1]) * np.sqrt(b[0] + b[1])
    cos_value = (c[0] + c[1]) / denom if denom != 0 else 1.0
    return cos_value


class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                OrderedDict.popitem(self, last=False)

    def get(self, key):
        res = OrderedDict.get(self, key)
        OrderedDict.move_to_end(self, key, last=True)
        return res


if __name__ == '__main__':
    a = LimitedSizeDict(size_limit=3)
    a[1] = "a"
    a[2] = "b"
    res = a.get(1)
    a[3] = "c"
    a[4] = "d"
    print()
