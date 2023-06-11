#ifndef SPARSEDAM_H
#define SPARSEDAM_H

#include <iostream>
using namespace std;
#include <unordered_map>
#include <vector>
#include "tsl/robin_map.h"
#include "Trajectory.h"


class SparseDAM {
public:
    int seg_num;
    int mask_size;
    int lb_csmv;

    vector<tsl::robin_map<int, int>> csm_row;
    vector<tsl::robin_map<int, int>> csm_col;

    SparseDAM();

    SparseDAM(int seg_num, int mask_size, int lb_csmv);

    void generate(vector<Trajectory>& data);

    int load(string& csm_path);

    void save(string& save_path);

    int get_csm_value(int o, int d);
    vector<int> get_col(int d);

    vector<pair<int, int>> get_rank_list(int o, int d);
};


#endif //SPARSEDAM_H
