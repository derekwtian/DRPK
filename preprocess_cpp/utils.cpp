#include "utils.h"

std::vector<std::string> split(std::string s, const std::string& delimiter)
{
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    if(int(s.length()) > 0)
    {
        tokens.push_back(s);
    }
    return tokens;
}

double calc_cos_value(const std::vector<double>& v1, const std::vector<double>& v2) {
    auto a0 = v1[0] * v1[0];
    auto a1 = v1[1] * v1[1];
    auto b0 = v2[0] * v2[0];
    auto b1 = v2[1] * v2[1];
    auto c0 = v1[0] * v2[0];
    auto c1 = v1[1] * v2[1];
    auto denom = sqrt(a0 + a1) * sqrt(b0 + b1);
    double cos_value = 1.0;
    if (denom != 0) {
        cos_value = (c0 + c1) / denom;
    }
    return cos_value;
}

std::vector<int> get_time_array(long long timestamp, int tz) {
    auto milli = timestamp + tz * 3600 * 1000; //此处转化为东八区北京时间，如果是其它时区需要按需求修改
    auto mTime = std::chrono::milliseconds(milli);
    auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(mTime);
    auto tt = std::chrono::system_clock::to_time_t(tp);
    thread_local struct tm time_arr{};
    gmtime_r(&tt, &time_arr);
    std::vector<int> a = {time_arr.tm_wday, time_arr.tm_hour, time_arr.tm_min, time_arr.tm_sec};
    return a;
}

std::pair<int, double> get_timeslot(std::vector<int>& time_arr, int time_delta) {
    int num_1h = 60 * 60 / time_delta;
    int num_1d = 24 * num_1h;
    int seconds = time_arr[2] * 60 + time_arr[3];
    int idx = ((time_arr[0] + 6) % 7) * num_1d + time_arr[1] * num_1h + seconds / time_delta;
    double residual = (seconds % time_delta) * 1.0 / time_delta;
    return std::make_pair(idx, residual);
}

int get_time_idx(std::vector<int>& time_arr) {
    int time_idx = time_arr[1];
    if (time_arr[0] == 0 || time_arr[0] == 6) {
        time_idx += 24;
    }
    return time_idx;
}