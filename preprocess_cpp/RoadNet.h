#ifndef ROADNET_H
#define ROADNET_H

#include <iostream>
using namespace std;
#include <vector>
#include <string>
#include "utils.h"
#include <fstream>
#include <numeric>
#include <sstream>
#include <iomanip>
#include "tsl/robin_map.h"


static int const CITY_SEG_NUM[] = {6786, 5699, 11491, 12383, 65276, 185074, 311769, 26659, 59927, 107655};
static int const TIMEZONES[] = {8, 8, 1, -7, 0, 1, 0, -7, 8, 8};


class Segment {
public:
    int eid;
    string src_id;
    string trg_id;
    float length;
    int road_type;
    vector<double> src_geo;
    vector<double> trg_geo;
    vector<double> seg_vec;
    float azimuth;
    int frequency = -1;
    float travel_time = -1;
    float travel_speed = -1;
    vector<int> adjList;

    Segment();
    Segment(string& id, string& uid, string& vid, string& length, string& rt, string& geo_u, string& geo_v, string& azimuth);

    void set_frequency(int freq);
    void set_speed(float speed);
    void set_adjList(const vector<int>& adj_ids);

};

class RoadNet {
public:
    int seg_num;
    vector<Segment> segs;

    RoadNet() {
        this->seg_num = -1;
    }

    explicit RoadNet(int seg_num) {
        this->seg_num = seg_num;
        this->segs.resize(seg_num);
    }

    int load(string& load_path);

    void add_freq_speed(vector<int>& unigram, vector<vector<float>>& speed);

    void save(string& save_path);

    float get_seg_length(int seg);

    vector<double> get_src_geo(int seg);

    vector<double> get_trg_geo(int seg);

    vector<double> get_seg_vec(int seg);

    double get_candidate_cos(int seg, int des);

    int open(string& seginfo_path);
    vector<int> get_neighbors(int eid);
};

#endif //ROADNET_H
