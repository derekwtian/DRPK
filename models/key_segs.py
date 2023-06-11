import math
import os
import datetime as dt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from preprocess import SegInfo, SparseDAM


class FeatureGenerator(object):
    def __init__(self, workspace, seg_num, mask_size, time_delta, utc):
        self.csm = SparseDAM(workspace, seg_num, mask_size)

        self.workspace = workspace
        self.seg_size = seg_num
        self.mask_size = mask_size
        self.PAD_ID = 0

        self.time_delta = time_delta
        self.num_1h = int(60 * 60 / self.time_delta)
        self.num_1d = 24 * self.num_1h
        self.seg_info = SegInfo(os.path.join(workspace, "seg_info.csv"))
        self.angle_delta = 30.0
        self.tz = dt.timezone(dt.timedelta(hours=utc))

        self.vehicle_num = np.load(os.path.join(workspace, "vehicle_num_{}-48.npy".format(time_delta)))

    def get_time_idx(self, timestamp):
        time_arr = dt.datetime.fromtimestamp(timestamp, self.tz)
        seconds = time_arr.minute * 60 + time_arr.second
        idx = time_arr.weekday() * self.num_1d + time_arr.hour * self.num_1h + seconds // self.time_delta
        residual = (seconds % self.time_delta) * 1.0 / self.time_delta
        return int(idx), residual

    def get_time_idx2(self, timestamp):
        time_arr = dt.datetime.fromtimestamp(timestamp, self.tz)
        if time_arr.weekday() in [0, 1, 2, 3, 4]:
            idx = time_arr.hour
        else:
            idx = time_arr.hour + 24
        t_r = (time_arr.minute * 60 + time_arr.second) * 1.0 / 3600
        return int(idx), t_r

    def load4ksd(self, phase):
        data = pd.read_csv(os.path.join(self.workspace, "{}_keysegs.txt".format(phase)), sep=" ", header=None, names=['o', 'd', 'timestamp', 'timeslot', 'p', 'offset', 'candi', 'candi_feat', 'keyseg'])

        output = []
        for i in range(data.shape[0]):
            sample = data.iloc[i]
            if sample['keyseg'] == "-1":
                continue

            keyseg_info = []
            for item in sample['keyseg'].split(';'):
                label = [int(item2) for item2 in item.split(',')]
                keyseg_info.append(label)
            if keyseg_info[0][2] >= self.mask_size:
                continue
            path = [int(item) for item in sample['p'].split(',')]
            num = int(math.ceil(len(path) * 0.2))
            keyseg_info = keyseg_info[:num]

            offset = [float(item) for item in sample['offset'].split(',')]
            offset = np.array(offset, dtype=np.float32)
            timeslot, _ = self.get_time_idx2(sample['timestamp'])
            candidates = [int(item) for item in sample['candi'].split(',')]
            candis_feat = []
            if len(candidates) < self.mask_size:
                candidates.extend((self.mask_size - len(candidates)) * [self.PAD_ID])
            candidates = np.array(candidates[:self.mask_size], dtype=np.int64)

            tmp = [int(sample['o']), int(sample['d']), offset, timeslot, keyseg_info, candidates, candis_feat]
            output.append(tmp)
        return output

    def get_timeslots_offsets(self, data):
        timeslots = []
        offsets = []
        for item in data:
            o = item[0]
            d = item[1]
            t = item[2]
            o_offset = item[3]
            d_offset = item[4]

            t_p, t_r = self.get_time_idx2(t)
            if o_offset > 0:
                o_offset = o_offset * 1.0 / self.seg_info.get_seg_length(o)
            if d_offset > 0:
                d_offset = d_offset * 1.0 / self.seg_info.get_seg_length(d)

            timeslots.append(t_p)
            offsets.append([t_r, o_offset, d_offset])
        return timeslots, offsets


class KeySegData(Dataset):
    def __init__(self, data, seg_size, mask_size):
        self.data = data
        self.seg_size = seg_size
        self.mask_size = mask_size

    def __getitem__(self, idx):
        src, dest, offset, t_p, key_info, candidates, candidates_feat = self.data[idx]
        label, distribution = self.get_onehot_target(key_info, candidates)
        return src, dest, offset, t_p, label, distribution, candidates, candidates_feat

    def __len__(self):
        return len(self.data)

    def get_onehot_target(self, key_info, candidates):
        if len(candidates) == 0 or len(key_info) == 0:
            target = np.zeros(self.mask_size, dtype=np.float32)
            distribution = np.zeros(self.mask_size, dtype=np.float32)
            return target, distribution

        keysegs = dict()
        for item in key_info:
            keysegs[item[0]] = item[1]
        target, distribution = [], []
        for item in candidates:
            if item in keysegs:
                target.append(1)
                distribution.append(keysegs[item])
            else:
                target.append(0)
                distribution.append(0)

        target = np.array(target, dtype=np.float32)
        distribution = np.array(distribution, dtype=np.float32)
        if distribution.sum() > 0:
            distribution = distribution / distribution.sum()
        return target, distribution


class MLP(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, dropout=0.1):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.BatchNorm1d(dim_hidden, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class ODEncoder(nn.Module):
    def __init__(self, hparams):
        super(ODEncoder, self).__init__()
        self.hparams = hparams
        # Node2Vec feature: [seg_num, 64]
        if self.hparams.use_node2vec:
            node2vec = torch.from_numpy(np.load(hparams.pretrained_input_emb_path))
            self.node2vec_feat = nn.Embedding.from_pretrained(node2vec, freeze=False)
            self.node2vec_dest = nn.Embedding.from_pretrained(node2vec, freeze=False)
        else:
            self.node2vec_feat = nn.Embedding(num_embeddings=hparams.seg_num, embedding_dim=hparams.d_seg)
            self.node2vec_dest = nn.Embedding(num_embeddings=hparams.seg_num, embedding_dim=hparams.d_seg)

    def forward(self, src, dest, offset, t):
        emb_o = self.node2vec_feat(src)
        emb_d = self.node2vec_dest(dest)
        code = torch.cat((emb_o, emb_d), dim=1)
        if self.hparams.use_offset:
            code = torch.cat((code, offset[:, 1:3]), dim=1)
        return code


class KeySegPred(nn.Module):
    def __init__(self, hparams):
        super(KeySegPred, self).__init__()
        self.od_layer = ODEncoder(hparams)
        # MLP2 with two layer
        self.mlp = MLP(hparams.d_s, hparams.d_m7, hparams.d_m8, hparams.dropout)

        dam_data = pd.read_csv(hparams.dam, sep=" ", header=None, names=['row', 'col', 'value']).to_numpy(dtype=np.int32)
        dam = torch.sparse_coo_tensor(indices=dam_data[:, :2].transpose(), values=dam_data[:, 2], size=(hparams.seg_num, hparams.seg_num), dtype=torch.float32, device=hparams.device)
        self.dam_row = dam.to_sparse_csr()
        self.dam_col = dam.t().to_sparse_csr()

        self.mask_size = hparams.mask_size
        self.seg_num = hparams.seg_num
        self.lb_csmv = 1

        segs_geo = torch.from_numpy(np.load(hparams.segs_geo))
        self.segs_src = segs_geo[:, :2].to(hparams.device)
        self.segs_trg = segs_geo[:, 2:].to(hparams.device)
        self.traffic_popularity = torch.from_numpy(np.load(hparams.traffic_popularity)).to(hparams.device)
        self.cosine_similarity = nn.CosineSimilarity(dim=2)
        self.use_traffic4key = hparams.use_traffic4key
        if self.use_traffic4key:
            self.candidate_layer = nn.Linear(2, 4)
            self.dense = nn.Linear(hparams.d_m8 - self.candidate_layer.out_features, hparams.seg_num)
        else:
            self.dense = nn.Linear(hparams.d_m8, hparams.seg_num)

    def get_candidates(self, src, dest):
        batch_size = src.shape[0]
        idx = torch.arange(batch_size, dtype=torch.int, device=src.device)
        ones = torch.ones(batch_size, dtype=torch.int, device=src.device)
        src_sparse = torch.sparse_coo_tensor(indices=torch.vstack([idx, src]), values=ones, size=(batch_size, self.seg_num), dtype=torch.float32, device=src.device).to_sparse_csr()
        des_sparse = torch.sparse_coo_tensor(indices=torch.vstack([idx, dest]), values=ones, size=(batch_size, self.seg_num), dtype=torch.float32, device=src.device).to_sparse_csr()
        src_rows = torch.matmul(src_sparse, self.dam_row).to_dense()
        des_cols = torch.matmul(des_sparse, self.dam_col).to_dense()

        ranks = torch.min(src_rows, des_cols)
        weights, candidates = torch.topk(ranks, k=self.mask_size, dim=1)
        candidates = candidates * (weights >= self.lb_csmv)
        exists = weights[:, 0] >= self.lb_csmv
        return candidates, exists

    def forward(self, src, dest, offset, t, candidates, train_phase=True, use_top1=False):
        vec_od = self.od_layer(src, dest, offset, t)
        output = self.mlp(vec_od)
        exists = np.ones(src.shape[0], dtype=np.int32)
        if candidates is None:
            candidates, exists = self.get_candidates(src, dest)
        sub_w_ = self.dense.weight[candidates]
        sub_b_ = self.dense.bias[candidates]
        if self.use_traffic4key:
            traffic_feat = (self.traffic_popularity[candidates] * F.one_hot(t.unsqueeze(-1).repeat(1, candidates.shape[1]), self.traffic_popularity.shape[1])).sum(dim=2)
            vec1 = self.segs_src[dest.unsqueeze(-1).repeat(1, candidates.shape[1])] - self.segs_src[candidates]
            vec2 = self.segs_trg[candidates] - self.segs_src[candidates]
            cos_feat = self.cosine_similarity(vec1, vec2)
            candidates_feat = torch.cat([cos_feat.unsqueeze(-1), traffic_feat.unsqueeze(-1)], dim=2).type(dtype=torch.float32)

            emb_candidate = self.candidate_layer(candidates_feat)
            sub_w_ = torch.cat((sub_w_, emb_candidate), dim=2)
        output = torch.matmul(sub_w_, output.unsqueeze(-1)).squeeze(-1) + sub_b_

        if train_phase:
            return output
        else:
            if use_top1:
                g_kseg = candidates[:, 0]
            else:
                g_prob, g_kseg = torch.max(output, dim=1)
                g_kseg = torch.sum(F.one_hot(g_kseg, output.shape[1]) * candidates, dim=1)
            return g_kseg, exists
