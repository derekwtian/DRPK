#include "RoadNet.h"

string geo_tostring(vector<double>& a) {
    stringstream ss;
    ss << setprecision(8);
    ss << a[0] << "," << a[1];
    return ss.str();
}

float get_speed_by_rt(int rt) {
    float speed = 15.0;
    switch (rt) {
        case 0:
            speed = 33.0;
            break;
        case 1:
            speed = 27.0;
            break;
        case 2:
            speed = 22.0;
            break;
        case 3:
            speed = 16.0;
            break;
        case 4:
            speed = 11.0;
            break;
        case 5:
            speed = 8.0;
            break;
        case 6:
            speed = 6.0;
            break;
        case 7:
            speed = 1.5;
            break;
    }
    return speed;
}

Segment::Segment() {
    this->eid = -1;
    this->length = -1;
    this->road_type = -1;
    this->azimuth = -1;
}

Segment::Segment(string& id, string& uid, string& vid, string& length, string& rt, string& geo_u, string& geo_v, string& azimuth) {
    this->eid = stoi(id);
    this->src_id = uid;
    this->trg_id = vid;
    this->length = stof(length);
    this->road_type = stoi(rt);
    this->azimuth = stof(azimuth);

    string delimiter = ",";
    vector<string> tmp;

    tmp = split(geo_u, delimiter);
    this->src_geo.push_back(stod(tmp[0]));
    this->src_geo.push_back(stod(tmp[1]));

    tmp = split(geo_v, delimiter);
    this->trg_geo.push_back(stod(tmp[0]));
    this->trg_geo.push_back(stod(tmp[1]));

    this->seg_vec.push_back(this->trg_geo[0] - this->src_geo[0]);
    this->seg_vec.push_back(this->trg_geo[1] - this->src_geo[1]);
}

void Segment::set_frequency(int freq) {
    this->frequency = freq;
}

void Segment::set_speed(float speed) {
    this->travel_speed = speed;
    this->travel_time = this->length / speed;
}

void Segment::set_adjList(const vector<int>& adj_ids) {
    this->adjList = adj_ids;
}


int RoadNet::load(string& load_path) {
    ifstream ifs;

    ifs.open(load_path, ios::in);
    if (!ifs.is_open()) {
        cout << "open seg_info failed"<< endl;
        return -1;
    }

    string field1, field2, field3, field4, field5, field6, field7, field8;
    int count = 0;
    while (ifs >> field1 && ifs >> field2 && ifs >> field3 && ifs >> field4 && ifs >> field5 && ifs >> field6 && ifs >> field7 && ifs >> field8) {
        Segment seg(field1, field2, field3, field4, field5, field6, field7, field8);
        this->segs[seg.eid] = seg;
        count += 1;
    }
    ifs.close();

    cout << "Number of Segments: " << count << endl;
    return count;
}

void RoadNet::add_freq_speed(vector<int>& unigram, vector<vector<float>>& speed) {
    for (int i = 0; i < unigram.size(); i++) {
        this->segs[i].set_frequency(unigram[i]);

        float seg_speed;
        if (speed[i].empty()) {
            seg_speed = get_speed_by_rt(this->segs[i].road_type);
        } else {
            auto sum = accumulate(speed[i].begin(), speed[i].end(), 0.0);
            auto mean =  sum / speed[i].size();
            if (mean < 1e-2) {
                mean = get_speed_by_rt(this->segs[i].road_type);
            }
            seg_speed = mean;
        }
        this->segs[i].set_speed(seg_speed);
    }
}

void RoadNet::save(string& save_path) {
    ofstream ofs;
    ofs.open(save_path, ios::out);
    int count = 0;
    for (auto& seg : this->segs) {
        stringstream ss;
        ss << setprecision(8);

        ss << seg.eid << " " << seg.src_id << " " << seg.trg_id << " " << seg.length << " " << seg.road_type << " " << geo_tostring(seg.src_geo) << " " << geo_tostring(seg.trg_geo) << " " << seg.azimuth << " " << seg.frequency << " " << seg.travel_time;

        ofs << ss.str() << endl;
    }
    ofs.close();
    cout << "Saved Seg_info Successful, Segments Number: " << this->seg_num << endl;
}

float RoadNet::get_seg_length(int seg) {
    return this->segs[seg].length;
}

vector<double> RoadNet::get_src_geo(int seg) {
    return this->segs[seg].src_geo;
}

vector<double> RoadNet::get_trg_geo(int seg) {
    return this->segs[seg].trg_geo;
}

vector<double> RoadNet::get_seg_vec(int seg) {
    return this->segs[seg].seg_vec;
}

double RoadNet::get_candidate_cos(int seg, int des) {
    auto seg_src = this->get_src_geo(seg);
    auto d_src = this->get_src_geo(des);
    vector<double> vec1 = {d_src[0] - seg_src[0], d_src[1] - seg_src[1]};
    auto vec2 = this->get_seg_vec(seg);
    return calc_cos_value(vec1, vec2);
}

int RoadNet::open(string &seginfo_path) {
    ifstream ifs;

    ifs.open(seginfo_path, ios::in);
    if (!ifs.is_open()) {
        cout << "open seg_info failed"<< endl;
        return -1;
    }

    tsl::robin_map<string, vector<int>> adjList;

    string field1, field2, field3, field4, field5, field6, field7, field8, field9, field10;
    int count = 0;
    while (ifs >> field1 && ifs >> field2 && ifs >> field3 && ifs >> field4 && ifs >> field5 && ifs >> field6 && ifs >> field7 && ifs >> field8 && ifs >> field9 && ifs >> field10) {
        Segment seg(field1, field2, field3, field4, field5, field6, field7, field8);
        seg.frequency = stoi(field9);
        seg.travel_time = stof(field10);
        seg.travel_speed = seg.length / seg.travel_time;

        this->segs[seg.eid] = seg;
        auto pos = adjList.find(seg.src_id);
        if (pos == adjList.end()) {
            adjList[seg.src_id] = vector<int>();
        }
        adjList[seg.src_id].emplace_back(seg.eid);
        count += 1;
    }
    ifs.close();

    // add typology info for each segment
    for (auto & seg : this->segs) {
        seg.set_adjList(adjList[seg.trg_id]);
    }

    cout << "Number of Segments in RoadNet: " << count << endl;
    return count;
}

vector<int> RoadNet::get_neighbors(int eid) {
    return this->segs[eid].adjList;
}
