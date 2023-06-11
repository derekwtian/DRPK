#ifndef PREPROCESSING_H_
#define PREPROCESSING_H_

#include <map>
#include "Trajectory.h"
#include "RoadNet.h"
#include "Traffic.h"
#include "ThreadPool.h"
#include "SparseDAM.h"

int load_trajectories(string& traj_path, vector<Trajectory>& data);

int select_keyseg(Trajectory& traj, map<int, pair<int, int>>& rank_map, set<int>& candi_set, int label_size);
void save_keyseg_info(vector<Trajectory>& trajs, string& file_path);
void save_keyseg_info(vector<Trajectory>& trajs, vector<Trajectory>& sub_trajs, string& file_path);

void generate_unigram_speeds(vector<Trajectory> & trajs, vector<int>& unigram, vector<vector<float>>& speed);

void calc_feature_each(Trajectory& traj, SparseDAM& csm, RoadNet& roadnet, Traffic& traffic, int utc, int time_delta);
int calc_feature_sparse(vector<Trajectory>* trajs, int left, int right, SparseDAM* csm, RoadNet* roadnet, Traffic* traffic, int utc, int time_delta);
void mp_calc_feature(int threadnum, vector<Trajectory>& data, SparseDAM& csm, RoadNet& roadnet, Traffic& traffic, int utc, int time_delta);

void keyseg_labeling_each(Trajectory& traj);
int keyseg_labeling(vector<Trajectory>* trajs, int left, int right);
void mp_keyseg_labeling(int threadnum, vector<Trajectory>& data);

void graph_node2vec(string & workspace, int seg_num, vector<Trajectory> & trajs);

#endif
