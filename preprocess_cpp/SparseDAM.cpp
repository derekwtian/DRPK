#include "SparseDAM.h"

class compareRank {
public:
    bool operator()(const pair<int, int>& a, const pair<int, int>& b) {
        bool flag = a.second > b.second;
        if (a.second == b.second) {
            flag = a.first < b.first;
        }
        return flag;
    }
};

SparseDAM::SparseDAM() {
    this->seg_num = -1;
    this->mask_size = -1;
    this->lb_csmv = -1;
}

SparseDAM::SparseDAM(int seg_num, int mask_size, int lb_csmv) {
    this->seg_num = seg_num;
    this->mask_size = mask_size;
    this->lb_csmv = lb_csmv;

    vector<tsl::robin_map<int, int>> row(seg_num,tsl::robin_map<int, int>());
    vector<tsl::robin_map<int, int>> col(seg_num,tsl::robin_map<int, int>());
    this->csm_row = row;
    this->csm_col = col;
}

void SparseDAM::generate(vector<Trajectory> &data) {
    int src, des;
    for (auto & traj : data) {
        vector<int>& path = traj.seg_seq;
        for (int i = 0; i < path.size(); i++) {
            for (int j = i+1; j < path.size(); j++) {
                src = path[i];
                des = path[j];
                this->csm_row[src][des]++;
                this->csm_col[des][src]++;
            }
        }
    }
}

void SparseDAM::save(string &save_path) {
    ofstream ofs;
    ofs.open(save_path, ios::out);
    int count = 0;
    for (int i=0; i<this->seg_num; i++) {
        for (auto & it : this->csm_row[i]) {
            ofs << i << " " << it.first << " " << it.second << endl;
            count++;
        }
    }
    ofs.close();
    cout << "Saved CSM Successful, Segments Number: " << this->seg_num << ", Non-zero number: " << count << endl;
}

int SparseDAM::load(string &csm_path) {
    ifstream ifs;
    ifs.open(csm_path, ios::in);
    if (!ifs.is_open()) {
        cout << "open csm_all failed"<< endl;
        return -1;
    }

    int count = 0;
    int row, col, value;
    while (ifs >> row && ifs >> col && ifs >> value) {
        this->csm_row[row][col] = value;
        this->csm_col[col][row] = value;
        count++;
    }
    ifs.close();
    cout << "Loaded, Segments Number: "<< this->seg_num << ", Non-zero number: " << count << endl;
    return 0;
}

int SparseDAM::get_csm_value(int o, int d) {
    auto pos = this->csm_row[o].find(d);
    if (pos != this->csm_row[o].end()) {
        return pos->second;
    } else {
        return 0;
    }
}

vector<pair<int, int>> SparseDAM::get_rank_list(int o, int d) {
    vector<pair<int, int>> candidates;
    auto &src = this->csm_row[o];
    auto &des = this->csm_col[d];
    for (auto & it : src) {
        int o2k = it.second;
        if (o2k >= this->lb_csmv) {
            auto pos = des.find(it.first);
            if (pos != des.end()) {
                int k2d = pos->second;
                int rank = k2d;
                if (rank > o2k) {
                    rank = o2k;
                }
                if (rank >= this->lb_csmv) {
                    pair<int, int> tmp(it.first, rank);
                    candidates.emplace_back(tmp);
                }
            }
        }
    }
    sort(candidates.begin(), candidates.end(), compareRank());
    if (this->mask_size > 0 && candidates.size() > this->mask_size) {
        candidates.resize(this->mask_size);
    }
    return candidates;
}

vector<int> SparseDAM::get_col(int d) {
    auto & des_col = this->csm_col[d];
    vector<int> res(this->seg_num, 0);
    for (const auto & it : des_col) {
        res[it.first] = it.second;
    }
    return res;
}
