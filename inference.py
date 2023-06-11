import argparse
import math
import multiprocessing
import os
import pickle

import pandas as pd
import numpy as np
import torch
from models import FeatureGenerator
from train_keyseg import load_model
from preprocess import load_road_graph, load_paths, calc_cos_value
import time
from conf import keyseg_para
from utils import dict_to_object


class RoutePlanner(object):
    def __init__(self, workspace, model_ckpt, device, max_seq_len=79, len_need_keyseg=3000, break_tie="all", debug=False):
        self.G = load_road_graph(os.path.join(workspace, "road_graph_wtime"))
        if model_ckpt is None:
            city = 'chengdu'
            if 'ptl' in workspace:
                city = 'porto_large'
            elif 'bjl' in workspace:
                city = 'beijing_large'
            elif 'cdl' in workspace:
                city = 'chengdu_large'
            elif 'xal' in workspace:
                city = 'xian_large'
            elif 'sfl' in workspace:
                city = 'sanfran_large'
            print("[Info] City: {}".format(city))
            hparams = dict_to_object(keyseg_para[city])
        else:
            self.model, hparams = load_model(model_ckpt, device)

        self.feats_generator = FeatureGenerator(workspace,
                                                seg_num=hparams.seg_num,
                                                mask_size=hparams.mask_size,
                                                time_delta=hparams.time_delta,
                                                utc=hparams.utc)
        self.vehicle_num = np.load(os.path.join(workspace, "vehicle_num_{}-48.npy".format(hparams.time_delta)))
        self.num_1d = self.vehicle_num.shape[1]
        self.device = device
        self.seg_num = self.feats_generator.seg_size
        self.mask_size = hparams.mask_size
        self.max_seq_len =max_seq_len
        self.len_need_keyseg = len_need_keyseg
        self.debug = debug
        self.workspace = workspace

        self.break_tie = break_tie
        self.freq_limit = 1
        self.dcsm_theta = 1
        self.PAD_ID = 0
        self.second_planning_time = 0

        if self.debug:
            self.step_total = 0
            self.step_tie = 0
            self.step_tie_ocsm = 0
            self.step_tie_angle = 0
            self.step_tie_traffic = 0

    def merge(self, gens_k1, gens_k2, ds):
        res = []
        for gen_k1, gen_k2, d in zip(gens_k1, gens_k2, ds):
            travel_time1 = self.feats_generator.seg_info.get_path_travel_time(gen_k1)
            travel_time2 = self.feats_generator.seg_info.get_path_travel_time(gen_k2)
            if (gen_k1[-1] == d and gen_k2[-1] == d) or (gen_k1[-1] != d and gen_k2[-1] != d):
                if travel_time1 < travel_time2:
                    res.append(gen_k1)
                else:
                    res.append(gen_k2)
            else:
                if gen_k1[-1] == d:
                    res.append(gen_k1)
                else:
                    res.append(gen_k2)
        return res

    def fetch_preds(self, os, ds, ts, pr_os, pr_ds, k, use_top1=False):
        start = time.time()
        multi_k = self.get_keyseg(os, ds, ts, pr_os, pr_ds, k, use_top1)

        if self.debug:
            print("    > Time of Predicting Key Segments for {} OD Pairs: {:.3f}s".format(len(ds), time.time() - start))

        start = time.time()
        gens = self.planning_multi_batch(multi_k[k], ts)

        second_time = time.time()
        if k >= 2:
            gens_k1 = self.planning_multi_batch(multi_k[k-1], ts)
            gens = self.merge(gens_k1, gens, ds)
        self.second_planning_time += time.time() - second_time

        if self.debug:
            print("    > Time of CSM Planning for {} OD Pairs: {:.3f}s".format(len(ds), time.time() - start))

        return gens, multi_k[k]

    def planning_multi_batch(self, ods, ts):
        if self.debug:
            self.stats_sp = [0, 0.0]
        start = time.time()
        preds = []
        for i, od in enumerate(ods):
            route = self.planning_multi(od, ts[i])
            preds.append(route)

        if self.debug:
            duration = time.time() - start
            steps = 0
            for item in preds:
                steps += len(item)
            print("      > Break Tie (SP) for {} preds: Count: {} = {} + {}, Time: {:.3f}s = {:.3f}s + {:.3f}s".format(len(preds), steps, steps - self.stats_sp[0], self.stats_sp[0], duration, duration - self.stats_sp[1], self.stats_sp[1]))
        return preds

    def planning_multi(self, od, t):
        pred = [od[0]]
        timestamp = t + self.feats_generator.seg_info.get_seg_travel_time(od[0])
        for i in range(len(od)-1):
            o = od[i]
            d = od[i+1]

            # extended path will not transitable
            if pred[-1] != o:
                break

            col_d = self.feats_generator.csm.get_col(d)
            route = [o]
            seg_used = np.zeros(self.seg_num, dtype=np.int32)
            seg_used[o] = 1
            seg_used[d] = 1

            while len(route) < self.max_seq_len and route[-1] != d:
                out_segs = list(self.G.neighbors(route[-1]))
                # dead road, cannot reach d
                if len(out_segs) == 0:
                    break

                # judge tie
                nextseg = -1
                next_max = -1
                tie_cnt = 0
                tie_nbrs = []
                for seg in out_segs:
                    # d in neighbors, directly select d
                    if seg == d:
                        nextseg = d
                        tie_cnt = 1
                        break

                    if seg_used[seg] >= self.freq_limit:
                        continue
                    curr_prob = col_d[seg]
                    if curr_prob is None:
                        curr_prob = 0

                    if curr_prob > next_max:
                        nextseg = seg
                        tie_cnt = 1
                        next_max = curr_prob
                        tie_nbrs = [seg]
                    elif curr_prob == next_max:
                        tie_cnt += 1
                        tie_nbrs.append(seg)

                if tie_cnt != 1:
                    if tie_cnt == 0:
                        tie_nbrs = out_segs

                    t0 = time.time()
                    if self.break_tie == 'all':
                        if next_max < self.dcsm_theta:
                            nextseg, _ = self.break_tie_angle(route[-1], tie_nbrs, d)
                        else:
                            nextseg, flag = self.break_tie_traffic_flow(tie_nbrs, timestamp)
                            if flag:
                                nextseg, _ = self.break_tie_angle(route[-1], tie_nbrs, d)
                    elif self.break_tie == 'wotp':
                        nextseg = self.break_tie_random(tie_nbrs)
                    else:
                        raise NotImplementedError

                    if self.debug:
                        self.stats_sp[1] += time.time() - t0
                        self.stats_sp[0] += 1
                        self.step_tie += 1

                if nextseg == -1:
                    break
                route.append(nextseg)
                timestamp += self.feats_generator.seg_info.get_seg_travel_time(nextseg)
                seg_used[nextseg] += 1
                if self.debug:
                    self.step_total += 1

            pred = pred + route[1:]
        return pred

    def break_tie_random(self, tie_nbrs):
        return tie_nbrs[0]

    def break_tie_angle(self, curr, tie_nbrs, d):
        curr_geo = self.feats_generator.seg_info.get_seg_geo(curr)
        curr_trg = curr_geo[2:]
        d_geo = self.feats_generator.seg_info.get_seg_geo(d)
        d_src = d_geo[:2]
        vec1 = d_src - curr_trg

        nextseg = -1
        next_max = -2
        tie_cnt = 0
        for seg in tie_nbrs:
            vec2 = self.feats_generator.seg_info.get_seg_vec(seg)
            cos_value = calc_cos_value(vec1, vec2)
            if cos_value > next_max:
                nextseg = seg
                tie_cnt = 1
                next_max = cos_value
            else:
                tie_cnt += 1

        flag = False
        return nextseg, flag

    def break_tie_traffic_flow(self, tie_nbrs, timestamp):
        idx, _ = self.feats_generator.get_time_idx2(timestamp)
        nextseg = -1
        next_max = -1
        tie_cnt = 0
        for seg in tie_nbrs:
            prob = self.vehicle_num[seg, idx]
            if prob > next_max:
                nextseg = seg
                tie_cnt = 1
                next_max = prob
            elif prob == next_max:
                tie_cnt += 1
        flag = False
        if tie_cnt > 1:
            flag = True
            if self.debug:
                self.step_tie_traffic += 1
        return nextseg, flag

    def get_odpair(self, ods):
        os = []
        ds = []
        pos = []
        for item in ods:
            max_dist = -1
            idx = 0
            for i in range(len(item)-1):
                dist = self.feats_generator.seg_info.get_od_dist(item[i], item[i+1])
                if dist > max_dist:
                    idx = i
                    max_dist = dist
            os.append(item[idx])
            ds.append(item[idx+1])
            pos.append(idx+1)
        return os, ds, pos

    def get_keyseg(self, oris, ds, ts, pr_os, pr_ds, k, use_top1):
        multi_k = []
        ods = list(zip(oris, ds))
        multi_k.append(ods)
        timeslots = []
        if k >= 1:
            timeslots, offsets = self.feats_generator.get_timeslots_offsets(list(zip(oris, ds, ts, pr_os, pr_ds)))
        factor = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device)
        for key_num in range(k):
            if key_num == 0:
                srcs = oris
                dests = ds
            else:
                srcs, dests, pos = self.get_odpair(ods)

            # blindly select the top-1 as keyseg
            self.model.eval()
            with torch.no_grad():
                src = torch.as_tensor(srcs, device=self.device)
                dest = torch.as_tensor(dests, device=self.device)
                offset = torch.as_tensor(offsets, device=self.device)
                t_vec = torch.as_tensor(timeslots, device=self.device)
                if key_num >= 1:
                    offset = offset * factor

                outputs, pool_empty = self.model(src, dest, offset, t_vec, None, False, use_top1=use_top1)

            key_seg = outputs.detach().cpu().tolist()
            rank_exist = pool_empty.detach()

            res = []
            for i in range(rank_exist.shape[0]):
                od = ods[i]
                if rank_exist[i] == 1:
                    o = srcs[i]
                    d = dests[i]
                    if key_num == 0:
                        res.append((o, key_seg[i], d))
                    else:
                        res.append(od[:pos[i]] + (key_seg[i],) + od[pos[i]:])
                else:
                    res.append(od)
            multi_k.append(res)
            ods = res
        return multi_k


def calc_metric(pred, ground, segs_info):
    if pred[-1] == ground[-1]:
        reach = 1
    else:
        reach = 0
    pred = set(pred)
    ground = set(ground)
    disjunction = pred & ground
    conjunction = pred | ground
    res = [len(disjunction), len(pred), len(ground), len(conjunction), reach]
    weighted_res = [0, 0, 0, 0, reach]
    for edge in disjunction:
        weighted_res[0] += segs_info[edge]['length']
    for edge in pred:
        weighted_res[1] += segs_info[edge]['length']
    for edge in ground:
        weighted_res[2] += segs_info[edge]['length']
    for edge in conjunction:
        weighted_res[3] += segs_info[edge]['length']
    return res, weighted_res


def metric_out(ratios, digits=4):
    df = pd.DataFrame(ratios, columns=['dis', 'gen', 'real', 'con', 'reach'])

    reachable = df['reach'].sum() * 1.0 / df.shape[0]
    unreachable = df.shape[0] - df['reach'].sum()
    print("Reachability: {}, Unreachable: {}".format(round(reachable, 4), unreachable))

    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    jaccard = 0.0
    for i in range(df.shape[0]):
        data = df.iloc[i]
        p_per = data['dis'] * 1.0 / data['gen']
        r_per = data['dis'] * 1.0 / data['real']
        if p_per < 1e-12 or r_per < 1e-12:
            f1_per = 0
        else:
            f1_per = 2 * p_per * r_per / (p_per + r_per)
        jac_per = data['dis'] * 1.0 / data['con']
        precision += p_per
        recall += r_per
        f1_score += f1_per
        jaccard += jac_per
    precision /= df.shape[0]
    recall /= df.shape[0]
    f1_score /= df.shape[0]
    jaccard /= df.shape[0]
    print("Effectiveness: {}".format([round(precision, digits), round(recall, digits), round(f1_score, digits), round(jaccard, digits)]))


def do_eval(preds, truths, nodes):
    print("==========Starting Evaluation==========")
    print("Path Number: {}".format(len(truths)))
    metric = []
    metric_weighted = []
    for pred, path in zip(preds, truths):
        res, res_weighted = calc_metric(pred, path, nodes)
        metric.append(res)
        metric_weighted.append(res_weighted)

    metric_out(metric_weighted, digits=3)


def get_odt(bat_data):
    paths = []
    os = []
    ds = []
    ts = []
    pr_os = []
    pr_ds = []
    for path, auxi in bat_data:
        paths.append(path)
        os.append(path[0])
        ds.append(path[-1])
        ts.append(auxi[0][1])
        pr_os.append(auxi[0][0])
        pr_ds.append(auxi[1][0])
    return os, ds, ts, pr_os, pr_ds, paths


def do_prediction(data, planner, k=1, batch_size=32, debug=False):
    print("==========Predicting...==========")
    path_num = len(data)
    batch_num = int(math.ceil(path_num * 1.0 / batch_size))
    print("==> path_num: {}, batch_size: {}, batch_num: {}".format(path_num, batch_size, batch_num))

    duration = 0
    pred = []
    real = []
    keyseg = []
    for i in range(batch_num):
        bat_data = data[i*batch_size: (i+1)*batch_size]
        os, ds, ts, pr_os, pr_ds, paths = get_odt(bat_data)

        t0 = time.time()
        gens, ods = planner.fetch_preds(os, ds, ts, pr_os, pr_ds, k)
        duration += time.time() - t0
        if debug:
            print("  > Time of Getting Predicted Routes for {} OD Pairs: {:.3f}s".format(len(paths), time.time() - t0))
        real.extend(paths)
        pred.extend(gens)
        keyseg.extend(ods)
    duration = duration
    qps = path_num * 1.0 / duration
    print("Number of trajs predicted per second (QPS): {} = {} / {}".format(round(qps, 2), path_num, duration))
    return pred, real, keyseg


def infer():
    parser = argparse.ArgumentParser(description='routeplanning.py')
    parser.add_argument('--workspace', type=str, default="/home/tianwei/Projects/ETA4RP/data/sfl_aug")
    parser.add_argument('--test_file', type=str, default="/home/tianwei/dataset/split_dataset/sanfran_large/traj_test")
    parser.add_argument('--model_path', type=str, default="/home/tianwei/Projects/ETA4RP/data/sfl_aug/model_keyseg_smlkp2_m100_bce10_od130_kfeat4")
    parser.add_argument('-gpu_id', type=str, default="0")
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-max_seq_len', type=int, default=300)
    parser.add_argument('-left', type=int, default=5)
    parser.add_argument('-right', type=int, default=300)
    parser.add_argument('-len_left', type=int, default=0)
    parser.add_argument('-len_right', type=int, default=100000)
    parser.add_argument('-len_need_keyseg', type=int, default=0)
    parser.add_argument('-key_num', type=int, default=1)
    parser.add_argument('-break_tie', type=str, choices=['all', 'wotp'], default="all")
    parser.add_argument("-threadnum", type=int, default=int(math.ceil(multiprocessing.cpu_count() * 0.6)))
    parser.add_argument('-debug', action="store_true", default=False)
    parser.add_argument("-save_preds", action="store_true", default=False)
    parser.add_argument("-cpu", action="store_true", dest="force_cpu")
    opt = parser.parse_args()
    if opt.key_num == 0:
        model_path = None
        save_path = opt.workspace
    else:
        model_path = os.path.join(opt.model_path, "model_acchigh.ckpt")
        save_path = opt.model_path
    print(opt)

    device = torch.device("cuda:{}".format(opt.gpu_id) if ((not opt.force_cpu) and torch.cuda.is_available()) else "cpu")
    print("running this on {}".format(device))
    planner = RoutePlanner(opt.workspace, model_path,
                           device=device,
                           max_seq_len=opt.max_seq_len,
                           len_need_keyseg=opt.len_need_keyseg,
                           break_tie=opt.break_tie,
                           debug=opt.debug)

    test_data, _ = load_paths(opt.test_file, left=opt.left, right=opt.right)

    test_data = [item for item in test_data if opt.len_left <= planner.feats_generator.seg_info.get_od_dist(item[0][0], item[0][-1]) < opt.len_right]
    print("Number of Paths in [{}, {}) will be Test: {}".format(opt.len_left, opt.len_right, len(test_data)))

    gens, truths, keysegs = do_prediction(test_data, planner, k=opt.key_num,
                                 batch_size=opt.batch_size,
                                 debug=opt.debug)
    if opt.debug:
        print("Tie Stats:", planner.step_total, planner.step_tie, planner.step_tie_ocsm, planner.step_tie_angle, planner.step_tie_traffic)
    if opt.save_preds:
        pickle.dump(list(zip(truths, gens, keysegs)), open(os.path.join(save_path, "truths_gens-{}-k{}.pkl".format(opt.break_tie, opt.key_num)), "wb"))
    do_eval(gens, truths, planner.G.nodes)
    print("[Info] Test k={} of Trajectories in [{}, {}) Finished.".format(opt.key_num, opt.len_left, opt.len_right))


if __name__ == '__main__':
    infer()
