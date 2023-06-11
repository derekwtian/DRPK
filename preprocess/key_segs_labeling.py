import csv
import math
import multiprocessing as mp
import os
from time import time

from .utils import load_paths
from .seg_info import SegInfo


def select_key_segs(real, rank_list):
    intersection = set(rank_list.keys()) & set(real[1:-1])
    if len(intersection) == 0:
        return [-1, []]

    disjunction = []
    for item in intersection:
        info = rank_list[item]
        disjunction.append((item, info[0], info[1]))
    if len(disjunction) > 1:
        disjunction.sort(key=lambda elem: elem[0])
        disjunction.sort(key=lambda elem: elem[1], reverse=True)

    top0 = disjunction[0][2]
    disjunction = disjunction[:len(rank_list)]
    return [top0, disjunction]


def label_key_segs(bat_data, opt, seg_num):
    from models import FeatureGenerator

    processor = FeatureGenerator(opt.workspace,
                                 seg_num=seg_num,
                                 mask_size=opt.mask_size,
                                 time_delta=opt.time_delta,
                                 utc=opt.utc)
    dam = processor.csm
    data_rows = []
    cnt = 0
    t_rank = 0
    t_inter = 0
    for i in range(len(bat_data)):
        path = bat_data[i][0]
        o = path[0]
        d = path[-1]
        t0 = time()
        attn = dam.get_rank_list(o, d)
        t_rank += time() - t0
        if len(attn) == 0:
            continue
        rank_list = {}
        for idx, item in enumerate(attn):
            rank_list[item[0]] = [item[1], idx+1]

        t0 = time()
        res = select_key_segs(path, rank_list)
        t_inter += time() - t0
        if res[0] == -1:
            continue
        cnt += 1
        offset = bat_data[i][1]
        timeslot, residual = processor.get_time_idx(offset[0][1])
        offset_tmp = [residual,
                      offset[0][0] / processor.seg_info.get_seg_length(o),
                      offset[1][0] / processor.seg_info.get_seg_length(d)]
        keysegs = [",".join(str(i) for i in item) for item in res[1]]
        data_rows.append([o, d, offset[0][1], timeslot,
                          ",".join([str(item) for item in path]),
                          ",".join([str(item) for item in offset_tmp]),
                          ",".join([str(item[0]) for item in attn[:opt.mask_size]]), -1, ";".join(keysegs)])
    return data_rows, cnt


def keyseg_labeling(opt, phase="train"):
    if phase == "train":
        data_path = opt.train_file
    elif phase == "valid":
        data_path = opt.valid_file
    elif phase == "test":
        data_path = opt.test_file
    else:
        print("Phase should be in train, valid and test")
        raise

    data, _ = load_paths(data_path, left=opt.left, right=opt.right)

    seginfo_file = os.path.join(opt.workspace, "seg_info.csv")
    seg_info = SegInfo(seginfo_file, cache_size=5000)

    path_num = len(data)
    print("Number of Paths will be Labeled: {}".format(path_num))
    chunksize = opt.chunk_size
    chunk_num = int(math.ceil(path_num * 1.0 / chunksize))
    print("==> chunk_size: {}, chunk_num: {}".format(chunksize, chunk_num))

    out_file = os.path.join(opt.workspace, "{}_keysegs.txt".format(phase))
    with open(out_file, 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=',')
        fields_output_file.writerows([])

    pool = mp.Pool(processes=opt.threadnum)
    results = []
    cnts = []
    print("Start labeling...")
    start_time = time()
    for i in range(chunk_num):
        sub_data = data[i*chunksize: (i+1)*chunksize]
        results.append(pool.apply_async(label_key_segs, args=(sub_data, opt, seg_info.seg_num)))
    pool.close()
    pool.join()
    print("Key Segment ({}) Finding Time: {:.3f}s".format(phase, time() - start_time))

    for p in results:
        rows, cnt = p.get()
        cnts.append(cnt)
        with open(out_file, 'a') as fp:
            fields_output_file = csv.writer(fp, delimiter=' ')
            fields_output_file.writerows(rows)

    print(path_num, sum(cnts))
    print("Saved {} successful".format(phase))


if __name__ == '__main__':
    pass
