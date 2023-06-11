from args import make_args
from preprocess import gen_dam, gen_node2vec_emb, keyseg_labeling, txt2npy, get_road_graph


if __name__ == '__main__':
    opt = make_args()
    print(opt)

    # 1 training CSM, seg_info, graph with time, gps feature
    print("==========1 Get CSM from train data.==========")
    if not opt.use_cpp:
        gen_dam(opt)
    txt2npy(opt.workspace)
    get_road_graph(opt.workspace)
    # 2 label key segs for training
    print("==========2 Label key segments for train, valid and test.==========")
    if not opt.use_cpp:
        keyseg_labeling(opt, "train")
        keyseg_labeling(opt, "valid")
        keyseg_labeling(opt, "test")
    # 3 node2vec emb
    print("==========3 Node2Vec Embedding for road graph.==========")
    gen_node2vec_emb(opt)

    print("Prepare Workspace Finished!")
