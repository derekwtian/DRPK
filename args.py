import math
import multiprocessing
from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()

    parser.add_argument('--workspace', type=str, default="data/porto_large")
    parser.add_argument('--train_file', type=str, default="porto_data/preprocessed_train_trips")
    parser.add_argument('--valid_file', type=str, default="porto_data/preprocessed_validation_trips")
    parser.add_argument('--test_file', type=str, default="porto_data/preprocessed_test_trips")
    parser.add_argument('--edges_shp', type=str, default="data/Porto_Large/edges.shp")
    parser.add_argument("-threadnum", type=int, default=int(math.ceil(multiprocessing.cpu_count() * 0.6)))
    parser.add_argument('-left', type=int, default=5)
    parser.add_argument('-right', type=int, default=300)

    # for node2vec
    parser.add_argument('-walk_len', type=int, default=30)
    parser.add_argument('-num_walks', type=int, default=25)
    parser.add_argument('-p', type=float, default=1.0)
    parser.add_argument('-q', type=float, default=1.0)
    parser.add_argument('-emb_dim', type=int, default=64)
    parser.add_argument('-window', type=int, default=5)
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-time_delta', type=int, default=3600)
    parser.add_argument('-utc', type=int, default=8)
    parser.add_argument("-neg", action="store_true", default=False)

    # for key segs labeling
    parser.add_argument('-label_size', type=int, default=30)
    parser.add_argument('-mask_size', type=int, default=100)
    parser.add_argument('-chunk_size', type=int, default=10000)
    parser.add_argument('-lb_damv', type=int, default=1)

    parser.add_argument('-use_cpp', action="store_true", default=False)

    args, unknown = parser.parse_known_args()
    if len(unknown)!= 0 and not args.ignore_unknown_args:
        print("some unrecognised arguments {}".format(unknown))
        raise SystemExit
    if args.neg:
        args.utc = 0 - args.utc

    return args
