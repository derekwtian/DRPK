#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include <iostream>
using namespace std;
#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <sstream>
#include "utils.h"

class Trajectory {
public:
    string oid;
    string tid;
    vector<double> offset_o;
    vector<double> offset_d;
    vector<int> seg_seq;
    set<int> path_set;
    vector<double> timestamps;
    vector<float> speeds;
    vector<float> travel_times;
    vector<vector<int>> keysegs;
    vector<pair<int, int>> rank_list;
    int top0 = -2;
    int od_value = 0;

    int feat_timeslot = -1;
    vector<double> feat_offset;
    vector<vector<double>> feat_candidate;

    Trajectory(string& oid, string& tid, string& str_o, string& str_d, string& segs, string& ts, string& str_speeds, string& str_traveltimes);

    Trajectory(string& oid, string& tid, vector<double>& offset_o, vector<double>& offset_d, vector<int>& segs, vector<double>& ts, vector<float>& speeds, vector<float>& travel_times);

    void set_keysegs(vector<vector<int>>& labels);

    void set_ranklist(vector<pair<int, int>>& candidates);

    void set_od_value(int od_csmv);

    string out_keyseg_string();

    string feat_offset_tostring();

    string candidate_tostring();

    string feat_candidate_tostring();

    string keyseg_tostring();

    string seg_seq_tostring();

    vector<Trajectory> gen_subtraj();

    friend ostream& operator <<(ostream& outputStream, const Trajectory& p) {
        outputStream << "oid = " << p.oid << ", tid = "<<p.tid;
        outputStream << endl << "seg_seq = ";
        for (int i=0; i<p.seg_seq.size(); i++) {
            outputStream << p.seg_seq[i] << " ";
        }
        return outputStream;
    }
};

#endif //TRAJECTORY_H
