#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <chrono>
#include <cmath>

std::vector<std::string> split(std::string s, const std::string& delimiter);

double calc_cos_value(const std::vector<double>& v1, const std::vector<double>& v2);

std::vector<int> get_time_array(long long timestamp, int tz);

std::pair<int, double> get_timeslot(std::vector<int>& time_arr, int time_delta);

int get_time_idx(std::vector<int>& time_arr);

#endif //UTILS_H
