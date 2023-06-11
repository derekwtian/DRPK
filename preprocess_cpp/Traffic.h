#ifndef TRAFFIC_H
#define TRAFFIC_H

#include <vector>
#include <climits>
#include "Trajectory.h"
#include "utils.h"


class Traffic {
public:
    int seg_num;
    int timeslot_num;
    int tz;
    std::vector<std::vector<int>> traffic_num;
    std::vector<int> min_traffic;
    std::vector<int> max_traffic;

    Traffic() {
        this->seg_num = -1;
        this->timeslot_num = -1;
        this->tz = 8;
    }

    Traffic(int seg_num, int timeslot_num, int tz) {
        this->seg_num = seg_num;
        this->timeslot_num = timeslot_num;
        this->tz = tz;
        std::vector<std::vector<int>> row(timeslot_num, std::vector<int>(seg_num, 0));
        this->traffic_num = row;
        this->min_traffic.resize(timeslot_num);
        this->max_traffic.resize(timeslot_num);
    }

    void generate(std::vector<Trajectory>& data) {
        int seg, time_idx;
        for (auto & traj : data) {
            auto& ts_seq = traj.timestamps;
            for (int i = 0; i < ts_seq.size(); i++) {
                if (traj.speeds[i] < 0.1 || traj.speeds[i] >= 35) {
                    continue;
                }
                auto time_arr = get_time_array(ts_seq[i] * 1000, this->tz);
                seg = traj.seg_seq[i];
                time_idx = get_time_idx(time_arr);
                this->traffic_num[time_idx][seg]++;
            }
        }
        set_min_max();
    }

    int load(string& load_path) {
        ifstream ifs;
        ifs.open(load_path, ios::in);
        if (!ifs.is_open()) {
            cout << "open traffic_num failed"<< endl;
            return -1;
        }

        int count = 0;
        int row, col, value;
        while (ifs >> row && ifs >> col && ifs >> value) {
            this->traffic_num[row][col] = value;
            count++;
        }
        ifs.close();
        cout << "Loaded, Segments Number: "<< this->seg_num << ", Total number: " << count << endl;
        set_min_max();
        return 0;
    }

    void save(string& save_path) {
        ofstream ofs;
        ofs.open(save_path, ios::out);
        int count = 0;
        for (int i = 0; i < this->timeslot_num; i++) {
            for (int j = 0; j < this->seg_num; j++) {
                ofs << i << " " << j << " " << this->traffic_num[i][j] << endl;
                count++;
            }
        }
        ofs.close();
        cout << "Saved Traffic_num Successful, Segments Number: " << this->seg_num << ", Total number: " << count << endl;
    }

    double get_traffic_norm(int t_idx, int seg) {
        return (this->traffic_num[t_idx][seg] - this->min_traffic[t_idx]) * 2.0 / (this->max_traffic[t_idx] - this->min_traffic[t_idx]) - 1;
    }

    int get_traffic_popularity(int t_idx, int seg) {
        return this->traffic_num[t_idx][seg];
    }

private:
    void set_min_max() {
        for (int i = 0; i < this->timeslot_num; i++) {
            int min = INT_MAX, max = -1;
            for (int j = 0; j < this->seg_num; j++) {
                if (this->traffic_num[i][j] < min) {
                    min = this->traffic_num[i][j];
                }
                if (this->traffic_num[i][j] > max) {
                    max = this->traffic_num[i][j];
                }
            }
            this->min_traffic[i] = min;
            this->max_traffic[i] = max;
        }
    }

};


#endif //TRAFFIC_H
