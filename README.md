# DRPK: Effective and Efficient Route Planning Using Historical Trajectories on Road Networks (VLDB 2023)

## Environment and Package Dependencies
The code can be successfully run under the following environments:
- Python: 3.8
- Pytorch version: 1.13 with CUDA 11.7
- OS: Ubuntu 20.04

The project will also need following package dependencies:
- numpy
- pandas
- networkx
- [microdict](https://github.com/touqir14/Microdict)
- haversine
- geopandas


## Format of the Data

Download the [preprocessed data](https://connectpolyu-my.sharepoint.com/:f:/g/personal/21037065r_connect_polyu_hk/EgvyOyo1eWNEjPcSjSsVM-0BQGVrfuA0NdTV8ocg6QsaJA?e=gGCXCf) and unzip the downloaded .zip file. For each city (San Francisco, Porto, Xi'an, Beijing, Chengdu), there are two types of data:

### 1. Map-matched trajectory data
The format of map-matched trajectory data like as follows.

```angular2html
edmugrip,1212038019:edmugrip,"[(137.96932188674592, 1212038019, -122.39606, 37.792731), (41.63089590260138, 1212038170, -122.40162, 37.793008)]","[[6557, 1212037972.113, 2.943, 71.205], [8965, 1212038043.318, 2.943, 15.766], [10763, 1212038059.085, 2.943, 3.908], [10761, 1212038062.993, 2.943, 34.729], [1607, 1212038097.721, 2.943, 36.551], [11780, 1212038134.272, 2.943, 72.254], [3634, 1212038206.526, 2.943, 36.875], [8612, 1212038243.402, 5.118, 55.304], [6290, 1212038298.705, 5.118, 20.926], [1610, 1212038319.631, 5.118, 20.414], [3402, 1212038340.045, 5.118, 20.371], [3404, 1212038360.416, 4.261, 24.524], [1612, 1212038384.94, 4.261, 24.546], [1614, 1212038409.486, 4.261, 34.233]]","[(-122.39636, 37.79236, 1212038019, 2.942585), (-122.3978, 37.79133, 1212038077, 5.11841), (-122.40007, 37.79343, 1212038137, 4.260853), (-122.40167, 37.79341, 1212038170, 4.260853)]"
...
```
Each line records four fields to represent a map-matched trajectory, i.e., moving object id, trajectory id, offsets and segment sequence. For example,

```angular2html
offsets: (137.96932188674592, 1212038019, -122.39606, 37.792731) means 
         (the distance between the start point of source segment and the source location,
          departure time,
          the longitude of source location,
          the latitude of source location,)
          The destination location has similar information.

segment sequence: [6557, 1212037972.113, 2.943, 71.205] means 
                  [segment id,
                   the timestamp when trajectory enter this segment,
                   the average speed when the trajectory go through this segment,
                   the duration for the trajectory through this segment]
```

The links for raw trajectory data used in DRPK are also provided as follows, you can also refer to our paper.
```
San Francisco: https://crawdad.org/epfl/mobility/20090224
Porto: https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data
Chengdu, Xi'an: https://outreach.didichuxing.com/research/opendata/
Beijing: obtained from a published paper and the data link is http://more.datatang.com/en
```

### 2. OSM map data
The maps used in DRPK is extracted from [OpenStreetMap](http://www.openstreetmap.org/export#). In the `map` folder, there are the following files:

1. `nodes.shp` : Contains OSM node information with unique `node id`.
2. `edges.shp` : Contains network connectivity information with unique `edge id`.


## Directory Structure
The directory structure is as follows.
```
codespace
    ├ args.py: the parameters for prepareing a workspace
    ├ conf.py: set the hyper-parameters in KSD model
    ├ prepare_workspace.py: generate DA indicator, Traffic popularity, and label the key segments for trajectories
    ├ train_keyseg.py: train KSD model
    ├ inference.py: predict routes for route planning queries
    ├ utils.py
    ├ models
    |   └ key_segs.py: KSD model
    └ preprocess
        ├ dam.py: construct DA indicator
        ├ seg_info.py: construct Traffic popularity
        ├ graph_embedding.py: gengrate node2vec embeddings
        └ key_segs_labeling.py: label the key segments for trajectory data
        
workspace (e.g., /data)
    └ dataset_name1 (e.g., porto_data)
        ├ map
        |   ├ nodes.shp
        |   └ edges.shp
        ├ traj_train.csv
        ├ traj_valid.csv
        └ traj_test.csv
```

## Usage
- Run `prepare_workspace.py` to generate the DA indicator, Traffic Popularity and label the historical trajectories for KSD model training. You can also modify `args.py` for the corresponding parameters when preparing your data workspace.
- Modify `conf.py` file to set the hyper-parameters of KSD model.
- Run main function in `train_keyseg.py` to train a KSD model.
- Run infer function in `inference.py` for online RPQs inference.

To get the details of the parameters when using DRP and DRPK, please refer to `args.py` and `conf.py`, respectively. Each field is described in detail in comments.

## Citations
If you use the code or data in this repository, citing our paper as the following will be really appropriate.
```
@article{tian2023drpk,
  author       = {Wei Tian and
                  Jieming Shi and
                  Siqiang Luo and
                  Hui Li and
                  Xike Xie and
                  Yuanhang Zou},
  title        = {Effective and Efficient Route Planning Using Historical Trajectories on Road Networks},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {16},
  number       = {10},
  pages        = {2512--2524},
  year         = {2023}
}
```
