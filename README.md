# ST-RMGCN

generate the semantic adjacency matrix：
    python genSimHash.py --save=Data/sensor_graph/metr-simHash8.npy
    
data pre-processing:
    python GenerateData.py

train the model:
    --python train.py --gcn_bool --addaptadj --randomadj --expid=metr-la-exp01
    
