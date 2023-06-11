import math
from ast import literal_eval
from time import time

import numpy as np
import pandas as pd
from haversine import haversine
import datetime as dt

from tqdm import tqdm

from .utils import LimitedSizeDict, load_paths2


rt_dict = {
    "motorway": 0,
    "trunk": 1,
    "primary": 2,
    "secondary": 3,
    "tertiary": 4,
    "unclassified": 5,
    "residential": 6,
    "motorway_link": 3,
    "trunk_link": 3,
    "primary_link": 3,
    "secondary_link": 4,
    "tertiary_link": 4,
    "living_street": 7,
}


def get_road_type(rt_str):
    avg = "secondary"
    if "[" in rt_str:
        rts = literal_eval(rt_str)
        codes = []
        for item in rts:
            if item not in rt_dict:
                codes.append(rt_dict[avg])
            else:
                codes.append(rt_dict[item])
        code = max(codes)
    else:
        if rt_str not in rt_dict:
            rt_str = avg
        code = rt_dict[rt_str]
    return code


def get_speed(rt):
    speed_info = {
        0: 33.0,
        1: 27.0,
        2: 22.0,
        3: 16.0,
        4: 11.0,
        5: 8.0,
        6: 6.0,
        7: 1.5,
    }
    return speed_info[rt]


def calc_azimuth(v1):
    '''
    :param v1: [begin_lon, begin_lat, end_lon, end_lat]
    :return: azimuth
    '''
    v2 = [v1[0], v1[1], v1[0], 90]
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = (angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = (angle2 * 180/math.pi)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    if v1[2] < v1[0]:
        included_angle = 360 - included_angle
    return included_angle


class SegInfo(object):
    def __init__(self, seginfo_file, cache_size=100000):
        segs = pd.read_csv(seginfo_file, sep=" ", header=None, names=['eid', 'src', 'trg', 'len', 'rt', 'geo_src', 'geo_trg', 'azimuth', 'freq', 'travel_time'])
        self.seg_num = segs.shape[0]
        self.__od_dist__ = LimitedSizeDict(size_limit=cache_size)
        self.__od_azimuth__ = LimitedSizeDict(size_limit=cache_size)
        self.__seg_info__ = np.zeros(self.seg_num, dtype=object)
        for i in range(segs.shape[0]):
            seg = segs.iloc[i]
            bp = [float(item) for item in seg['geo_src'].split(",")]
            ep = [float(item) for item in seg['geo_trg'].split(",")]
            tmp = (
                float(seg['len']),
                float(seg['travel_time']),
                int(seg['rt']),
                float(seg['azimuth']),
                np.array(bp + ep, dtype=float),
                np.array(ep, dtype=float) - np.array(bp, dtype=float)
            )
            self.__seg_info__[seg['eid']] = tmp

    def get_seg_length(self, seg):
        return self.__seg_info__[seg][0]

    def get_seg_travel_time(self, seg):
        return self.__seg_info__[seg][1]

    def get_seg_rt(self, seg):
        return self.__seg_info__[seg][2]

    def get_seg_azimuth(self, seg):
        return self.__seg_info__[seg][3]

    def get_seg_geo(self, seg):
        return self.__seg_info__[seg][4]

    def get_seg_vec(self, seg):
        return self.__seg_info__[seg][5]

    def get_path_distance(self, path):
        dist = 0.0
        for seg in path:
            dist += self.__seg_info__[seg][0]
        return dist

    def get_path_travel_time(self, path):
        tt = 0.0
        for seg in path:
            tt += self.__seg_info__[seg][1]
        return tt

    def get_od_dist(self, o, d):
        key = (o, d)
        if key in self.__od_dist__:
            res = self.__od_dist__.get(key)
        else:
            point1 = self.__seg_info__[o][4][2:]
            point2 = self.__seg_info__[d][4][:2]
            res = haversine((point1[1], point1[0]), (point2[1], point2[0]), unit="m")
            self.__od_dist__[key] = res
        return res

    def get_od_azimuth(self, o, d):
        key = (o, d)
        if key in self.__od_azimuth__:
            res = self.__od_azimuth__.get(key)
        else:
            o_ep = self.__seg_info__[o][4][2:].tolist()
            d_bp = self.__seg_info__[d][4][:2].tolist()
            res = calc_azimuth(o_ep + d_bp)
            self.__od_azimuth__[key] = res
        return res


def seginfo_constructor(edges, speeds, freqs):
    data = []
    for i in range(edges.shape[0]):
        tmp = edges.iloc[i]
        start = tmp['geometry'].coords[0]
        end = tmp['geometry'].coords[-1]
        azimuth = calc_azimuth([float(start[0]), float(start[1]), float(end[0]), float(end[1])])
        eid = tmp['fid']
        length = tmp['length']
        rt = get_road_type(tmp['highway'])
        speed = speeds[eid]
        if speed < 1e-2:
            speed = get_speed(rt)
        travel_time = length * 1.0 / speed
        data.append([eid,
                     tmp['u'],
                     tmp['v'],
                     round(length, 3),
                     rt,
                     "{},{}".format(start[0], start[1]),
                     "{},{}".format(end[0], end[1]),
                     azimuth,
                     freqs[eid],
                     round(travel_time, 3)])
    return data


def gen_vehicle_num(args, seg_num):
    data = load_paths2(args.train_file, left=args.left, right=args.right)
    tz = dt.timezone(dt.timedelta(hours=args.utc))
    num_1d = 24
    vehicle_num = np.zeros((seg_num, num_1d*2), dtype=np.int32)

    start_time = time()
    for traj in tqdm(data, desc="traj num"):
        for seg, ts, speed, _ in traj:
            if speed < 0.1 or speed >= 35:
                continue
            tm = dt.datetime.fromtimestamp(ts, tz)
            if tm.weekday() in [0, 1, 2, 3, 4]:
                idx = tm.hour
            else:
                idx = tm.hour + num_1d
            vehicle_num[seg, idx] += 1
    print("Traffic Popularity Time:" + '{:.3f}s'.format(time() - start_time))

    res = []
    for i in range(num_1d*2):
        for j in range(seg_num):
            res.append([i, j, vehicle_num[j, i]])
    return res


if __name__ == '__main__':
    pass
