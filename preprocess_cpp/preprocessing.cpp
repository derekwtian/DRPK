#include "preprocessing.h"

class compareKeySeg {
public:
    bool operator()(const vector<int>& a, const vector<int>& b) {
        return a[2] < b[2];
    }
};

int select_keyseg(Trajectory& traj, map<int, pair<int, int>>& rank_map, set<int>& candi_set, int label_size) {
    set<int>& path_set = traj.path_set;
    set<int> intersection;
    set_intersection(path_set.begin(), path_set.end(), candi_set.begin(), candi_set.end(), inserter(intersection, intersection.begin()));
    if (intersection.empty()) {
        return -1;
    }

    vector<vector<int>> disjunction;
    for (auto it = intersection.begin(); it != intersection.end(); it++) {
        auto value = rank_map.at(*it);
        vector<int> tmp;
        tmp.push_back(*it);
        tmp.push_back(value.first);
        tmp.push_back(value.second);
        disjunction.push_back(tmp);
    }
    sort(disjunction.begin(), disjunction.end(), compareKeySeg());
    int top0 = disjunction[0][2];
    if (disjunction.size() > label_size) {
        disjunction.resize(label_size);
    }
    traj.set_keysegs(disjunction);
    return top0;
}

int load_trajectories(string& traj_path, vector<Trajectory>& data) {
    data.clear();

    ifstream ifs;
    ifs.open(traj_path, ios::in);
    if (!ifs.is_open()) {
        cout << "open trajs failed"<< endl;
        return -1;
    }

    string field1, field2, field3, field4, field5, field6, field7, field8;
    int count = 0;
    while (ifs >> field1 && ifs >> field2 && ifs >> field3 && ifs >> field4 && ifs >> field5 && ifs >> field6 && ifs >> field7 && ifs >> field8) {
        Trajectory traj(field1, field2, field3, field4, field5, field6, field7, field8);
        data.push_back(traj);
        count += 1;
    }
    ifs.close();

    cout << "Number of Trajectories: " << count << endl;
    return count;
}

void save_keyseg_info(vector<Trajectory>& trajs, string& file_path) {
    int cnt0 = 0, cnt1 = 0, cnt2 = 0;
    ofstream ofs;
    ofs.open(file_path, ios::out);
    for (auto & traj : trajs) {
        ofs << traj.out_keyseg_string() << endl;
        if (traj.rank_list.empty()) {
            cnt1++;
        } else if (traj.keysegs.empty()) {
            cnt2++;
        } else {
            cnt0++;
        }
    }
    ofs.close();
    cout << "write number: " << trajs.size() << " = " << cnt0 << " + "  << cnt1 << " + "  << cnt2 << endl;
}

void save_keyseg_info(vector<Trajectory>& trajs, vector<Trajectory>& sub_trajs, string& file_path) {
    int cnt0 = 0, cnt1 = 0, cnt2 = 0, cnt4 = 0, cnt5 = 0, cnt6 = 0;
    ofstream ofs;
    ofs.open(file_path, ios::out);
    for (auto & traj : trajs) {
        ofs << traj.out_keyseg_string() << endl;
        if (traj.rank_list.empty()) {
            cnt1++;
        } else if (traj.keysegs.empty()) {
            cnt2++;
        } else {
            cnt0++;
        }
    }
    for (auto & traj : sub_trajs) {
        ofs << traj.out_keyseg_string() << endl;
        if (traj.rank_list.empty()) {
            cnt5++;
        } else if (traj.keysegs.empty()) {
            cnt6++;
        } else {
            cnt4++;
        }
    }
    ofs.close();
    cout << "write number: " << trajs.size() << " = " << cnt0 << " + "  << cnt1 << " + "  << cnt2 << endl;
    cout << "sub trajectories: " << sub_trajs.size() << " = " << cnt4 << " + "  << cnt5 << " + "  << cnt6 << endl;
}

void generate_unigram_speeds(vector<Trajectory> & trajs, vector<int>& unigram, vector<vector<float>>& speed) {
    int seg;
    for (auto traj : trajs) {
        vector<int>& path = traj.seg_seq;
        for (int i = 0; i < path.size(); i++) {
            seg = path[i];
            unigram[seg]++;
            if (traj.speeds[i] >= 0.1) {
                speed[seg].push_back(traj.speeds[i]);
            }
        }
    }
}

void calc_feature_each(Trajectory& traj, SparseDAM& csm, RoadNet& roadnet, Traffic& traffic, int utc, int time_delta) {
    int o = traj.seg_seq[0];
    int d = traj.seg_seq[traj.seg_seq.size() - 1];

    auto time_arr = get_time_array(traj.offset_o[1] * 1000, utc);
    auto timeslot = get_timeslot(time_arr, time_delta);

    traj.feat_timeslot = timeslot.first;
    traj.feat_offset.push_back(timeslot.second);
    auto res_o = traj.offset_o[0] / roadnet.get_seg_length(o);
    auto res_d = traj.offset_d[0] / roadnet.get_seg_length(d);
    traj.feat_offset.push_back(res_o);
    traj.feat_offset.push_back(res_d);

    auto rank_list = csm.get_rank_list(o, d);
    if (rank_list.empty()) {
        return;
    }
    traj.set_ranklist(rank_list);
    traj.set_od_value(csm.get_csm_value(o, d));
}

int calc_feature_sparse(vector<Trajectory>* trajs, int left, int right, SparseDAM* csm, RoadNet* roadnet, Traffic* traffic, int utc, int time_delta) {
    for (int i = left; i < right; i++) {
        calc_feature_each((*trajs)[i], *csm, *roadnet, *traffic, utc, time_delta);
    }
    return left;
}

void mp_calc_feature(int threadnum, vector<Trajectory>& data, SparseDAM& csm, RoadNet& roadnet, Traffic& traffic, int utc, int time_delta) {
    int chunk_size = ceil(data.size() * 1.0 / threadnum);
    cout<< "Trajectories Number: " << data.size() << ", Chunk_size: " << chunk_size << ", Thread Number: " << threadnum << endl;
    ThreadPool pool(threadnum);
    vector<future<int> > results;

    int left, right;
    for (int i = 0; i < threadnum; i++) {
        left = i * chunk_size;
        right = (i+1) * chunk_size;
        if (right > data.size()) {
            right = data.size();
        }
        results.emplace_back(
                pool.enqueue(calc_feature_sparse, &data, left, right, &csm, &roadnet, &traffic, utc, time_delta)
        );
    }

    for(auto && result: results)
        auto a = result.get();
}

void keyseg_labeling_each(Trajectory& traj) {
    if (traj.rank_list.empty()) {
        return;
    }

    map<int, pair<int, int>> rank_map;
    set<int> candi_set;
    for (int j = 0; j < traj.rank_list.size(); j++) {
        rank_map.insert(pair<int, pair<int, int>>(traj.rank_list[j].first, pair<int, int>(traj.rank_list[j].second, j+1)));
        candi_set.insert(traj.rank_list[j].first);
    }

    select_keyseg(traj, rank_map, candi_set, traj.rank_list.size());
}

int keyseg_labeling(vector<Trajectory>* trajs, int left, int right) {
    for (int i = left; i < right; i++) {
        keyseg_labeling_each((*trajs)[i]);
    }
    return left;
}

void mp_keyseg_labeling(int threadnum, vector<Trajectory>& data) {
    int chunk_size = ceil(data.size() * 1.0 / threadnum);
    cout<< "Trajectories Number: " << data.size() << ", Chunk_size: " << chunk_size << ", Thread Number: " << threadnum << endl;
    ThreadPool pool(threadnum);
    vector<future<int> > results;

    int left, right;
    for (int i = 0; i < threadnum; i++) {
        left = i * chunk_size;
        right = (i+1) * chunk_size;
        if (right > data.size()) {
            right = data.size();
        }
        results.emplace_back(
                pool.enqueue(keyseg_labeling, &data, left, right)
        );
    }

    for(auto && result: results)
        auto a = result.get();
}

void graph_node2vec(string &workspace, int seg_num, vector<Trajectory> &trajs) {
    string seginfo_path = workspace + "seg_info.csv";
    string save_path = workspace + "weighted_edges.txt";

    RoadNet map(seg_num);
    map.open(seginfo_path);
    tsl::robin_map<int, tsl::robin_map<int, int>> graph;
    for (auto & seg : map.segs) {
        for (auto id : seg.adjList) {
            graph[seg.eid][id] = 1;
        }
    }
    for (auto & traj : trajs) {
        auto & path = traj.seg_seq;
        for (int i=0; i<path.size()-1; i++) {
            graph[path[i]][path[i+1]] += 1;
        }
    }
    ofstream ofs;
    ofs.open(save_path, ios::out);
    int count = 0;
    for (const auto & it : graph) {
        for (const auto & it2 : it.second) {
            ofs << it.first << " " << it2.first << " " << it2.second << endl;
            count++;
        }
    }
    ofs.close();
    cout << "Saved Weighted_graph Successful, Edges Number: " << seg_num << ", Total number: " << count << endl;
}
