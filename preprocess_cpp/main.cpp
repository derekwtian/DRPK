#include <iostream>
using namespace std;
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <cmath>
#include "getopt.h"
#include <chrono>
#include <filesystem>
#include "ThreadPool.h"
#include "Trajectory.h"
#include "Traffic.h"
#include "RoadNet.h"
#include "SparseDAM.h"
#include "preprocessing.h"


int main(int argc, char * argv[]) {
    string workspace = "data/sanfran_100/";
    int city_idx = 3;
    string traj_dir = "data/sanfran_100/";
    string scale = "100";
    int mask_size = 200;
    int lb_csmv = 1;
    int thread_num = 4;
    int time_delta = 3600;
    bool use_subtraj = false;

    static struct option long_options[] = {
            {"workspace", required_argument, nullptr, 1},
            {"city_idx", required_argument, nullptr, 7},
            {"traj_dir", required_argument, nullptr, 2},
            {"scale", required_argument, nullptr, 6},
            {"mask_size", required_argument, nullptr, 3},
            {"lb_csmv", required_argument, nullptr, 5},
            {"thread_num", required_argument, nullptr, 8},
            {"time_delta", required_argument, nullptr, 9},
            {"use_subtraj", required_argument, nullptr, 10},
            {nullptr, 0, nullptr, 0}
    };
    static char* const short_options=(char *)"";

    int option_index = 0;
    int ret;
    while((ret = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
        switch (ret) {
            case 1:
                workspace = optarg;
                break;
            case 2:
                traj_dir = optarg;
                break;
            case 3:
                mask_size = stoi(optarg);
                break;
            case 5:
                lb_csmv = stoi(optarg);
                break;
            case 6:
                scale = optarg;
                break;
            case 7:
                city_idx = stoi(optarg);
                break;
            case 8:
                thread_num = stoi(optarg);
                break;
            case 9:
                time_delta = stoi(optarg);
                break;
            case 10:
                use_subtraj = stoi(optarg);
                break;
            case 0:
                break;
        }
    }

    cout << "Input Parameters: " << endl;
    cout << "\tworkspace: " << workspace << endl;
    cout << "\tcity: " << city_idx << endl;
    cout << "\ttraj_dir: " << traj_dir << endl;
    cout << "\tscale: " << scale << endl;
    cout << "\tmask_size: " << mask_size << endl;
    cout << "\tlb_csmv: " << lb_csmv << endl;
    cout << "\ttime_delta: " << time_delta << endl;
    cout << "\tuse_subtraj: " << use_subtraj << endl;
    cout << "\tthread_num: " << thread_num << endl;

    int seg_num = CITY_SEG_NUM[city_idx];
    string csm_path = workspace + "csm_all.txt";
    string traffic_num_path = workspace + "traffic_num.txt";
    string edgeinfo_path = workspace + "edges.csv";
    string seginfo_path = workspace + "seg_info.csv";
    string train_traj_path = traj_dir + "serialized_traj_train_" + scale;
    string valid_traj_path = traj_dir + "serialized_traj_valid";
    string test_traj_path = traj_dir + "serialized_traj_test";
    string train_keyseg_path = workspace + "train_keysegs.txt";
    string valid_keyseg_path = workspace + "valid_keysegs.txt";
    string test_keyseg_path = workspace + "test_keysegs.txt";

    auto start_time = chrono::system_clock::now();
    chrono::duration<double> elapsed_time = chrono::system_clock::now() - start_time;

    vector<Trajectory> trajs;
    // load train trajectories
    load_trajectories(train_traj_path, trajs);

    // construct road network
    vector<int> unigram(CITY_SEG_NUM[city_idx], 0);
    vector<vector<float>> speed_info(CITY_SEG_NUM[city_idx], vector<float>());
    generate_unigram_speeds(trajs, unigram, speed_info);
    RoadNet roadnet(CITY_SEG_NUM[city_idx]);
    roadnet.load(edgeinfo_path);
    roadnet.add_freq_speed(unigram, speed_info);
    if (!filesystem::exists(seginfo_path)) {
        roadnet.save(seginfo_path);
    }

    // construct vehicle number
    Traffic traffic_info(seg_num, 48, TIMEZONES[city_idx]);
    if (filesystem::exists(traffic_num_path)) {
        traffic_info.load(traffic_num_path);
    } else {
        start_time = chrono::system_clock::now();
        traffic_info.generate(trajs);
        elapsed_time = chrono::system_clock::now() - start_time;
        cout << "Traffic_num Counting Time (seconds): " << elapsed_time.count() << endl;
        traffic_info.save(traffic_num_path);
    }

    // construct graph for node2vec
    graph_node2vec(workspace, CITY_SEG_NUM[city_idx], trajs);

    // construct sparse dam
    SparseDAM csm(seg_num, mask_size, lb_csmv);
    if (filesystem::exists(csm_path)) {
        csm.load(csm_path);
    } else {
        start_time = chrono::system_clock::now();
        csm.generate(trajs);
        elapsed_time = chrono::system_clock::now() - start_time;
        cout << "Sparse DAM Generation Time (seconds): " << elapsed_time.count() << endl;
        csm.save(csm_path);
    }

    start_time = chrono::system_clock::now();
    mp_calc_feature( thread_num, trajs, csm, roadnet, traffic_info, TIMEZONES[city_idx], time_delta);
    elapsed_time = chrono::system_clock::now() - start_time;
    cout << "Features (train) Calculating Time (seconds): " << elapsed_time.count() << endl;
    start_time = chrono::system_clock::now();
    mp_keyseg_labeling( thread_num, trajs);
    elapsed_time = chrono::system_clock::now() - start_time;
    cout << "Key Segment (train) Finding Time (seconds): " << elapsed_time.count() << endl;
    if (!use_subtraj) {
        save_keyseg_info(trajs, train_keyseg_path);
    } else {
        vector<Trajectory> sub_trajs;
        for (auto & traj : trajs) {
            auto tmp = traj.gen_subtraj();
            for (auto & sub_traj : tmp) {
                sub_trajs.emplace_back(sub_traj);
            }
        }
        start_time = chrono::system_clock::now();
        mp_calc_feature( thread_num, sub_trajs, csm, roadnet, traffic_info, TIMEZONES[city_idx], time_delta);
        elapsed_time = chrono::system_clock::now() - start_time;
        cout << "Features (sub_train) Calculating Time (seconds): " << elapsed_time.count() << endl;
        start_time = chrono::system_clock::now();
        mp_keyseg_labeling( thread_num, sub_trajs);
        elapsed_time = chrono::system_clock::now() - start_time;
        cout << "Key Segment (sub_train) Finding Time (seconds): " << elapsed_time.count() << endl;
        save_keyseg_info(trajs, sub_trajs, train_keyseg_path);
    }

    load_trajectories(valid_traj_path, trajs);
    start_time = chrono::system_clock::now();
    mp_calc_feature( thread_num, trajs, csm, roadnet, traffic_info, TIMEZONES[city_idx], time_delta);
    elapsed_time = chrono::system_clock::now() - start_time;
    cout << "Features (valid) Calculating Time (seconds): " << elapsed_time.count() << endl;
    start_time = chrono::system_clock::now();
    mp_keyseg_labeling(thread_num, trajs);
    elapsed_time = chrono::system_clock::now() - start_time;
    cout << "Key Segment (valid) Finding Time (seconds): " << elapsed_time.count() << endl;
    save_keyseg_info(trajs, valid_keyseg_path);

    load_trajectories(test_traj_path, trajs);
    start_time = chrono::system_clock::now();
    mp_calc_feature( thread_num, trajs, csm, roadnet, traffic_info, TIMEZONES[city_idx], time_delta);
    elapsed_time = chrono::system_clock::now() - start_time;
    cout << "Features (test) Calculating Time (seconds): " << elapsed_time.count() << endl;
    start_time = chrono::system_clock::now();
    mp_keyseg_labeling(thread_num, trajs);
    elapsed_time = chrono::system_clock::now() - start_time;
    cout << "Key Segment (test) Finding Time (seconds): " << elapsed_time.count() << endl;
    save_keyseg_info(trajs, test_keyseg_path);

    return 0;
}
