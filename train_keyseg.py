import argparse
import os
import pickle
import random
import sys
import time

import psutil
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import FeatureGenerator, KeySegData, KeySegPred
from conf import keyseg_para
from utils import dict_to_object
import numpy as np


def cal_acc(pred, onehot_target):
    # obtain the seg id for testing
    pred = pred.softmax(dim=1)
    pred_seg = torch.argmax(pred, dim=1)

    # calculate Acc1
    batch_size = pred.shape[0]
    pred_seg_onehot = F.one_hot(pred_seg, onehot_target.shape[1])
    inner = torch.sum(pred_seg_onehot * onehot_target, dim=1)
    n_correct = torch.gt(inner, 0).sum().item()
    n_word = batch_size
    return n_correct, n_word


def cal_loss_BCE(outputs, onehot_target, weight):
    # sigmoid over (o, d, seg) dimension
    m = nn.Sigmoid()
    pred = m(outputs)

    loss_fn = nn.BCELoss(weight=weight)
    bce_loss = loss_fn(pred, onehot_target)
    return bce_loss


def cal_weight(distribution, base=10):
    # weight of positive samples scale up
    weight = torch.pow(base, distribution)
    return weight


def epoch_forward(data, model, device):
    o, d, offset, t, label, distribution, candidates, candidates_feat = data
    o = o.to(device, non_blocking=True)
    d = d.to(device, non_blocking=True)
    offset = offset.to(device, non_blocking=True)
    t = t.to(device, non_blocking=True)
    label = label.to(device, non_blocking=True)
    distribution = distribution.to(device, non_blocking=True)
    candidates = candidates.to(device, non_blocking=True)
    weight = cal_weight(distribution)

    outputs = model(o, d, offset, t, candidates, train_phase=True)
    loss = cal_loss_BCE(outputs, label, weight)

    n_correct, n_word = cal_acc(outputs, label)
    return loss, n_correct, n_word


def eval_epoch(model, valid_data, device):
    model.eval()
    total = 0
    right = 0
    total_loss = 0
    batch_num = 0
    with torch.no_grad():
        for data in valid_data:
            loss, n_correct, n_word = epoch_forward(data, model, device)

            total_loss += loss.item()
            right += n_correct
            total += n_word
            batch_num += 1

    acc1 = right * 1.0 / total
    loss_per_word = total_loss / batch_num
    return loss_per_word, acc1


def train(model, train_loader, valid_loader, device, opt, hparams):
    tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

    cpu_ram_records = []

    lr = opt.lr_base
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=hparams.lr_step,
                                                           factor=hparams.lr_decay, threshold=1e-3)

    best_epoch = 0
    best_epoch_accu = 0
    best_epoch_train = 0
    best_epoch_accu_train = 0
    total_train_step = 0
    train_accus = []
    train_losses = []
    valid_accus = []
    valid_losses = []
    trained_epoch = 0
    start_time = time.time()
    for epoch_i in range(opt.epochs):
        print("=========Epoch: {}=========".format(epoch_i))
        trained_epoch += 1
        model.train()
        total_train_loss = 0
        right = 0
        total = 0
        batch_num = 0
        for data in train_loader:
            loss, n_correct, n_word = epoch_forward(data, model, device)

            total_train_loss += loss.item()
            right += n_correct
            total += n_word
            batch_num += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_train_step += 1
            # write loss to tensorboard
            tb_writer.add_scalars("Train_loss", {'loss': loss.item()}, total_train_step)

        if epoch_i == 0:
            print('[Info] Single epoch time cost:{}'.format(time.time() - start_time))
        train_loss = round(total_train_loss / batch_num, 4)
        train_accu = round(right * 1.0 / total, 4)
        train_losses += [train_loss]
        train_accus += [train_accu]

        print("==> Evaluation")
        valid_loss, valid_accu = eval_epoch(model, valid_loader, device)

        valid_losses += [round(valid_loss, 4)]
        valid_accus += [round(valid_accu, 4)]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'params': dict(hparams), 'model': model.state_dict()}

        if train_loss <= min(train_losses):
            best_epoch_train = epoch_i
            torch.save(checkpoint, os.path.join(opt.output_dir, 'model_train.ckpt'))
            print('    - [Info] The checkpoint file (Train Loss Low) has been updated.')
        if train_accu >= max(train_accus):
            best_epoch_accu_train = epoch_i
            torch.save(checkpoint, os.path.join(opt.output_dir, 'model_train_acchigh.ckpt'))
            print('    - [Info] The checkpoint file (Train Acc High) has been updated.')
        if round(valid_loss, 4) <= min(valid_losses):
            best_epoch = epoch_i
            torch.save(checkpoint, os.path.join(opt.output_dir, 'model.ckpt'))
            print('    - [Info] The checkpoint file (Loss Low) has been updated.')
        if round(valid_accu, 4) >= max(valid_accus):
            best_epoch_accu = epoch_i
            torch.save(checkpoint, os.path.join(opt.output_dir, 'model_acchigh.ckpt'))
            print('    - [Info] The checkpoint file (Acc High) has been updated.')

        tb_writer.add_scalars('Loss', {'train': total_train_loss / batch_num, 'val': valid_loss}, epoch_i)
        tb_writer.add_scalars('Acc1', {'train': right * 1.0 / total, 'val': valid_accu}, epoch_i)
        tb_writer.add_scalar('learning_rate', lr, epoch_i)

        cpu_ram = psutil.Process(os.getpid()).memory_info().rss
        gpu_ram = torch.cuda.memory_stats(device=device)['active_bytes.all.current']
        cpu_ram_records.append(cpu_ram)
        tb_writer.add_scalar('cpu_ram', round(cpu_ram * 1.0 / 1024 / 1024, 3), epoch_i)
        tb_writer.add_scalar('gpu_ram', round(gpu_ram * 1.0 / 1024 / 1024, 3), epoch_i)

        scheduler.step(valid_accu)
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']

        if lr <= 0.9 * 1e-5:
            print("==> [Info] Early Stop since lr is too small After Epoch {}.".format(epoch_i))
            break

    print("[Info] Training Finished, using {:.3f}s for {} epochs".format(time.time() - start_time, trained_epoch))
    tb_writer.close()
    print("[Info] Train Loss lowest epoch: {}, loss: {}, acc1: {}".format(best_epoch_train,
                                                                          train_losses[best_epoch_train],
                                                                          train_accus[best_epoch_train]))
    print("[Info] Train Acc1 highest epoch: {}, loss: {}, acc1: {}".format(best_epoch_accu_train,
                                                                           train_losses[best_epoch_accu_train],
                                                                           train_accus[best_epoch_accu_train]))
    print("[Info] Validation Loss lowest epoch: {}, loss: {}, acc1: {}".format(best_epoch, valid_losses[best_epoch],
                                                                               valid_accus[best_epoch]))
    print("[Info] Validation Acc1 highest epoch: {}, loss: {}, acc1: {}".format(best_epoch_accu,
                                                                                valid_losses[best_epoch_accu],
                                                                                valid_accus[best_epoch_accu]))

    model_size = sys.getsizeof(model.parameters())
    print("model size: {} Bytes".format(model_size))
    gpu_ram = torch.cuda.memory_stats(device=device)['active_bytes.all.peak']
    print("peak gpu memory usage: {:.3f} MB".format(gpu_ram * 1.0 / 1024 / 1024))
    cpu_ram_peak = max(cpu_ram_records)
    print("current memory usage: {:.3f} MB".format(cpu_ram_peak * 1.0 / 1024 / 1024))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default="data/sfl_100")
    parser.add_argument('--output_dir', type=str, default="data/sfl_100/model_keyseg")
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-lr_base', type=float, default=1e-3)
    parser.add_argument('-gpu_id', type=str, default="0")
    parser.add_argument('--training_file', type=str, default="train_keysegs.txt")
    parser.add_argument('-city', type=str,
                        choices=['porto_large', 'beijing_large', 'chengdu_large', 'xian_large', 'sanfran_large'], default='sanfran_large')
    parser.add_argument("-cpu", action="store_true", dest="force_cpu")
    opt = parser.parse_args()
    print(opt)

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    device = torch.device(
        "cuda:{}".format(opt.gpu_id) if ((not opt.force_cpu) and torch.cuda.is_available()) else "cpu")
    print("running this on {}".format(device))

    hparams = dict_to_object(keyseg_para[opt.city])
    hparams.pretrained_input_emb_path = os.path.join(opt.workspace, hparams.pretrained_input_emb_path)
    hparams.segs_geo = os.path.join(opt.workspace, hparams.segs_geo)
    hparams.traffic_popularity = os.path.join(opt.workspace, hparams.traffic_popularity)
    hparams.dam = os.path.join(opt.workspace, hparams.dam)
    hparams.d_s = 2 * hparams.d_seg
    if hparams.use_offset:
        hparams.d_s += 2
    hparams.device = device
    print(hparams)

    # ========= Loading Dataset ========= #
    processor = FeatureGenerator(opt.workspace,
                                 seg_num=hparams.seg_num,
                                 mask_size=hparams.mask_size,
                                 time_delta=hparams.time_delta,
                                 utc=hparams.utc)
    t0 = time.time()
    train_data = processor.load4ksd("train")

    train_data = KeySegData(train_data, processor.seg_size, processor.mask_size)
    print("loading training data use {:.3f}s".format(time.time() - t0))

    train_ods = set()
    for item in train_data:
        o = item[0]
        d = item[1]
        train_ods.add((o, d))
    pickle.dump(train_ods, open(os.path.join(opt.output_dir, "train_ods.pkl"), "wb"))

    t0 = time.time()
    valid_data = processor.load4ksd("valid")

    valid_data = KeySegData(valid_data, processor.seg_size, processor.mask_size)
    print("loading validation data use {:.3f}s".format(time.time() - t0))

    print("Training Size: {}, Validation Size: {}".format(len(train_data), len(valid_data)))
    train_loader = DataLoader(dataset=train_data, batch_size=hparams.batch_size,
                              shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=hparams.batch_size,
                              shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    model = KeySegPred(hparams).to(device)

    train(model, train_loader, valid_loader, device, opt, hparams)
    print("[Info] Model Training Finished!")

    t0 = time.time()
    test_data = processor.load4ksd("test")

    test_data = KeySegData(test_data, processor.seg_size, processor.mask_size)
    print("loading test data use {:.3f}s".format(time.time() - t0))

    print("[Info] Test Starting...")
    print("=====> AccHigh, Training")
    model_testing(device,
                  model_path=os.path.join(opt.output_dir, 'model_acchigh.ckpt'),
                  test_data=train_data)
    print("=====> AccHigh, Test")
    model_testing(device,
                  model_path=os.path.join(opt.output_dir, 'model_acchigh.ckpt'),
                  test_data=test_data)


def load_model(model_file, device):
    checkpoint = torch.load(model_file, map_location=device)
    hparams = dict_to_object(checkpoint['params'])
    hparams.device = device

    model = KeySegPred(hparams).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model, hparams


def model_testing(device, model_path, test_data):
    model, hparams = load_model(model_path, device)
    print(hparams)

    print("Test Size: {}".format(len(test_data)))
    test_loader = DataLoader(dataset=test_data, batch_size=hparams.batch_size,
                             shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    loss, accu = eval_epoch(model, test_loader, device)
    print("loss: {}, acc1: {}".format(loss, accu))


if __name__ == '__main__':
    main()
