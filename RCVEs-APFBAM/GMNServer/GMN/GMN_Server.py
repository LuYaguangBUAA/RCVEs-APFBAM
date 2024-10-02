from evaluation import compute_similarity, auc
from loss import pairwise_loss, triplet_loss
from utils import *
from configure import *
import numpy as np
import torch.nn as nn
import collections
import time
import os
import torch
from flask import Flask, request, jsonify




app = Flask(__name__)

 
node_feature_type_dic = {
    '__label__' : 13,
    '__id__' : 14,
    '__label_id__' : 27,
    '__label_angle__' : 15,
    '__id_angle__' : 16,
    '__label_id_angle__' : 29
    }

checkpoints_dic = {
    '__label__' : './GMN_Models/label_GMN.pth',
    '__id__' : './GMN_Models/id_GMN.pth',
    '__label_id__' : './GMN_Models/label_id_GMN.pth',
    '__label_angle__' : './GMN_Models/label_angle_GMN.pth',
    '__id_angle__' : './GMN_Models/id_angle_GMN_S9.pth',
    '__label_id_angle__' : './GMN_Models/label_id_angle_GMN.pth'
    }


spe_checkpoints_dic = {

    }


# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

node_feature_dim = 13 + 14 + 2 #由训练特征维度决定 13 14 2  13label 14id 2angle
edge_feature_dim = 4
'''
model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)
model.load_state_dict(torch.load(checkpoint_test15))
model.eval() 
'''

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json()
    
    node_feature_type_get = data.get('nodeFeatureType', None)
    scene_name_get = data.get('sceneName', None)
    
    node_features_get = data.get('nodeFeatures', None)
    edge_features_get = data.get('edgeFeatures', None)    
    from_idx_get = data.get('fromIdx', None)
    to_idx_get = data.get('toIdx', None)
    graph_idx_get = data.get('graphIdx', None)
    n_nodes_get = data.get('nNodes', None)
    
    if data is None:
        return jsonify({'error': 'Missing required key: test'}), 400
    else:
        print("Get data")

    node_feature_type = node_feature_type_get
    scene_name = scene_name_get
    
    print('Node_feature_type:' + str(node_feature_type) + ' Scene_name:' + str(scene_name))
    
    if node_feature_type in node_feature_type_dic:
        node_feature_dim = node_feature_type_dic[node_feature_type]
    else:
        print(f"'{node_feature_type}' do not in Node Feature Dic.")
        
    if node_feature_type == '__id_angle__' and scene_name in spe_checkpoints_dic:
        checkpoint = spe_checkpoints_dic[scene_name]
    elif node_feature_type in checkpoints_dic:
        checkpoint = checkpoints_dic[node_feature_type]
    else:
        print(f"'{node_feature_type}' do not in Checkpoints Dic.")
        
        

    node_features = torch.tensor(node_features_get)
    edge_features = torch.tensor(edge_features_get)
    from_idx = torch.tensor(from_idx_get)
    to_idx = torch.tensor(to_idx_get)
    graph_idx = torch.tensor(graph_idx_get)

    n_nodes = n_nodes_get
    
    #print("node_features", node_features.shape, "edge_features", edge_features.shape, "from_idx", from_idx.shape, "to_idx", to_idx.shape, "graph_idx", graph_idx.shape)


    config = get_default_config()
    model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval() 

    t_start = time.time()
    with torch.no_grad():        
        eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device), to_idx.to(device), graph_idx.to(device), n_nodes)
    
    
    x, y = reshape_and_split_tensor(eval_pairs, 2)

    accumulated_pair_auc = []

    similarity = compute_similarity(config, x, y) 
    
    result = similarity.detach().numpy()

    
    print("result:", result.shape)
    
    
    print('time %.2fs' % ( time.time() - t_start)) 
    
    return jsonify({"result": result.tolist()})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5800)


