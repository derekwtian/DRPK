import time


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst=Dict()
    for k,v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst


def temporal_feat(timestamp):
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
    week = timeArray.tm_wday
    hour = timeArray.tm_hour
    minute = timeArray.tm_min
    return [week, hour, minute]


if __name__ == '__main__':
    feat = temporal_feat(1476059163.167)
    print(feat)
