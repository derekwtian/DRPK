#include "Trajectory.h"

Trajectory::Trajectory(string& oid, string& tid, string& str_o, string& str_d, string& segs, string& ts, string& str_speeds, string& str_traveltimes) {
    this->oid = oid;
    this->tid = tid;

    string delimiter = ",";
    vector<string> tmp;

    tmp = split(str_o, delimiter);
    for (int i=0; i<tmp.size(); i++) {
        this->offset_o.push_back(stod(tmp[i]));
    }
    tmp = split(str_d, delimiter);
    for (int i=0; i<tmp.size(); i++) {
        this->offset_d.push_back(stod(tmp[i]));
    }
    tmp = split(segs, delimiter);
    for (int i=0; i<tmp.size(); i++) {
        this->seg_seq.push_back(stoi(tmp[i]));
        if (i > 0 && i < tmp.size() - 1) {
            this->path_set.insert(stoi(tmp[i]));
        }
    }
    tmp = split(ts, delimiter);
    for (int i=0; i<tmp.size(); i++) {
        this->timestamps.push_back(stod(tmp[i]));
    }
    tmp = split(str_speeds, delimiter);
    for (int i=0; i<tmp.size(); i++) {
        this->speeds.push_back(stof(tmp[i]));
    }
    tmp = split(str_traveltimes, delimiter);
    for (int i=0; i<tmp.size(); i++) {
        this->travel_times.push_back(stof(tmp[i]));
    }
}

void Trajectory::set_keysegs(vector<vector<int>>& labels) {
    this->keysegs = labels;
    this->top0 = labels[0][2];
}

void Trajectory::set_ranklist(vector<pair<int, int>>& candidates) {
    this->rank_list = candidates;
}

void Trajectory::set_od_value(int od_csmv) {
    this->od_value = od_csmv;
}

string Trajectory::out_keyseg_string() {
    string output = to_string(this->seg_seq[0]) + " " + to_string(this->seg_seq[this->seg_seq.size()-1]) + " " + to_string(this->offset_o[1]) + " " + to_string(this->feat_timeslot);
    output += " " + seg_seq_tostring();
    output += " " + feat_offset_tostring();
    output += " " + candidate_tostring();
    output += " " + feat_candidate_tostring();
    output += " " + keyseg_tostring();
    return output;
}

string Trajectory::feat_offset_tostring() {
    ostringstream oss;
    oss.precision(8);
    oss.setf(std::ios::fixed);

    oss.str("");
    oss << this->feat_offset[0];
    for (int i = 1; i < this->feat_offset.size(); i++) {
        oss << "," << this->feat_offset[i];
    }
    return oss.str();
}

string Trajectory::candidate_tostring() {
    if (this->rank_list.empty()) {
        return "-1";
    }
    pair<int, int>& tmp = this->rank_list[0];
    string a = to_string(tmp.first);
    for (int i = 1; i< this->rank_list.size(); i++) {
        a += "," + to_string(this->rank_list[i].first);
    }
    return a;
}

string Trajectory::feat_candidate_tostring() {
    if (this->feat_candidate.empty()) {
        return "-1";
    }

    ostringstream oss;
    oss.precision(8);
    oss.setf(std::ios::fixed);

    oss.str("");
    vector<double>& tmp = this->feat_candidate[0];
    oss << tmp[0] << "," << tmp[1];
    for (int i = 1; i < this->feat_candidate.size(); i++) {
        oss << ";" << this->feat_candidate[i][0] << "," << this->feat_candidate[i][1];
    }
    return oss.str();
}

string Trajectory::keyseg_tostring() {
    if (this->keysegs.empty()) {
        return "-1";
    }
    vector<int>& tmp = this->keysegs[0];
    string a = to_string(tmp[0]) + "," + to_string(tmp[1]) + "," + to_string(tmp[2]);
    for (int i = 1; i < this->keysegs.size(); i++) {
        a += ";" + to_string(this->keysegs[i][0]) + "," + to_string(this->keysegs[i][1]) + "," + to_string(this->keysegs[i][2]);
    }
    return a;
}

string Trajectory::seg_seq_tostring() {
    string a = to_string(this->seg_seq[0]);
    for (int i = 1; i < this->seg_seq.size(); i++) {
        a += "," + to_string(this->seg_seq[i]);
    }
    return a;
}

vector<Trajectory> Trajectory::gen_subtraj() {
    vector<Trajectory> res;

    if (!this->keysegs.empty()) {
        int k0 = this->keysegs[0][0];
        int idx = find(this->seg_seq.begin(), this->seg_seq.end(), k0) - this->seg_seq.begin();
        vector<double> offset_key = {0, this->timestamps[idx]};

        vector<int> sub_seq1(this->seg_seq.begin(), this->seg_seq.begin() + idx + 1);
        vector<double> sub_ts1(this->timestamps.begin(), this->timestamps.begin() + idx + 1);
        vector<float> sub_speed1(this->speeds.begin(), this->speeds.begin() + idx + 1);
        vector<float> sub_travel1(this->travel_times.begin(), this->travel_times.begin() + idx + 1);

        vector<int> sub_seq2(this->seg_seq.begin() + idx, this->seg_seq.end());
        vector<double> sub_ts2(this->timestamps.begin() + idx, this->timestamps.end());
        vector<float> sub_speed2(this->speeds.begin() + idx, this->speeds.end());
        vector<float> sub_travel2(this->travel_times.begin() + idx, this->travel_times.end());

        Trajectory sub1(this->oid, this->tid, this->offset_o, offset_key, sub_seq1, sub_ts1, sub_speed1, sub_travel1);
        Trajectory sub2(this->oid, this->tid, offset_key, this->offset_d, sub_seq2, sub_ts2, sub_speed2, sub_travel2);

        if (sub_seq1.size() >= 3) {
            res.emplace_back(sub1);
        }
        if (sub_seq2.size() >= 3) {
            res.emplace_back(sub2);
        }
    }

    return res;
}

Trajectory::Trajectory(string &oid, string &tid, vector<double> &offset_o, vector<double> &offset_d, vector<int> &segs,
                       vector<double> &ts, vector<float> &speeds, vector<float> &travel_times) {
    this->oid = oid;
    this->tid = tid;
    this->offset_o = offset_o;
    this->offset_d = offset_d;

    this->seg_seq = segs;
    for (int i = 1; i < this->seg_seq.size() - 1; i++) {
        this->path_set.insert(this->seg_seq[i]);
    }
    this->timestamps = ts;
    this->speeds = speeds;
    this->travel_times = travel_times;
}
