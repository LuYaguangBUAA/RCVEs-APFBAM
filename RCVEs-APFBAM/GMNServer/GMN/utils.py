import collections
from dataset import GraphEditDistanceDataset, FixedGraphEditDistanceDataset, GraphEditDistanceDataset_OneHot, FixedGraphEditDistanceDataset_OneHot
from graphembeddingnetwork import GraphEmbeddingNet, GraphEncoder, GraphAggregator
from graphmatchingnetwork import GraphMatchingNet
import copy
import torch
import random
import os
import numpy as np
import networkx as nx

GraphData = collections.namedtuple('GraphData', [
    'from_idx',
    'to_idx',
    'node_features',
    'edge_features',
    'graph_idx',
    'n_graphs'])


def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
      tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
      n_splits: int, number of splits to split the tensor into.

    Returns:
      splits: a list of `n_splits` tensors.  The first split is [tensor[0],
        tensor[n_splits], tensor[n_splits * 2], ...], the second split is
        [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """
    feature_dim = tensor.shape[-1]
    tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split


def build_model(config, node_feature_dim, edge_feature_dim):
    """Create model for training and evaluation.

    Args:
      config: a dictionary of configs, like the one created by the
        `get_default_config` function.
      node_feature_dim: int, dimensionality of node features.
      edge_feature_dim: int, dimensionality of edge features.

    Returns:
      tensors: a (potentially nested) name => tensor dict.
      placeholders: a (potentially nested) name => tensor dict.
      AE_model: a GraphEmbeddingNet or GraphMatchingNet instance.

    Raises:
      ValueError: if the specified model or training settings are not supported.
    """
    config['encoder']['node_feature_dim'] = node_feature_dim
    config['encoder']['edge_feature_dim'] = edge_feature_dim

    encoder = GraphEncoder(**config['encoder'])
    aggregator = GraphAggregator(**config['aggregator'])
    if config['model_type'] == 'embedding':
        model = GraphEmbeddingNet(
            encoder, aggregator, **config['graph_embedding_net'])
    elif config['model_type'] == 'matching':
        model = GraphMatchingNet(
            encoder, aggregator, **config['graph_matching_net'])
    else:
        raise ValueError('Unknown model type: %s' % config['model_type'])

    optimizer = torch.optim.Adam((model.parameters()),
                                 lr=config['training']['learning_rate'], weight_decay=1e-5)

    return model, optimizer


def build_datasets(config):
    """Build the training and evaluation datasets."""
    config = copy.deepcopy(config)

    if config['data']['problem'] == 'graph_edit_distance':
        dataset_params = config['data']['dataset_params']
        validation_dataset_size = dataset_params['validation_dataset_size']
        del dataset_params['validation_dataset_size']
        # training_set = GraphEditDistanceDataset(**dataset_params)
        training_set = GraphEditDistanceDataset_OneHot(**dataset_params)
        dataset_params['dataset_size'] = validation_dataset_size
        # validation_set = FixedGraphEditDistanceDataset(**dataset_params)
        validation_set = FixedGraphEditDistanceDataset_OneHot(**dataset_params)
    else:
        raise ValueError('Unknown problem type: %s' % config['data']['problem'])
 
    return training_set, validation_set
        


def get_graph(batch):
    if len(batch) != 2:
        # if isinstance(batch, GraphData):
        graph = batch
        node_features = torch.from_numpy(graph.node_features)
        edge_features = torch.from_numpy(graph.edge_features)
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        return node_features, edge_features, from_idx, to_idx, graph_idx
    else:
        graph, labels = batch
        node_features = torch.from_numpy(graph.node_features)
        edge_features = torch.from_numpy(graph.edge_features)
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        labels = torch.from_numpy(labels).long()
    return node_features, edge_features, from_idx, to_idx, graph_idx, labels


'''
test_set_num: 测试数据数量
test_feature_nclass: 测试特征维度
'''
def get_test_once_graph(node_feature_dim, edge_feature_dim):
    
    n_nodes_range=[3, 3]
    p_edge_range=[1, 1]
    n_min, n_max = n_nodes_range
    p_min, p_max = p_edge_range
    n_nodes = np.random.randint(n_min, n_max + 1)
    p_edge = np.random.uniform(p_min, p_max)
    
    g = nx.erdos_renyi_graph(n_nodes, p_edge)
    
    
    
    
    batch_graphs = []
    batch_graphs.append((g, g))
    
    Graphs = []
    
    from_idx = []
    to_idx = []
    graph_idx = []
    
    for graph in batch_graphs:
        for inergraph in graph:
            Graphs.append(inergraph)
    
    n_total_nodes = 0
    n_total_edges = 0
    
    for i, g in enumerate(Graphs):
        n_nodes_ = g.number_of_nodes()
        n_edges_ = g.number_of_edges()
        edges = np.array(g.edges(), dtype=np.int32)
        # shift the node indices for the edges
        from_idx.append(edges[:, 0] + n_total_nodes)
        to_idx.append(edges[:, 1] + n_total_nodes)
        graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)
        n_total_nodes += n_nodes_
        n_total_edges += n_edges_


    one_hot_inds = np.random.randint(low=0, high=node_feature_dim, size=n_total_nodes)
    node_features = np.eye(node_feature_dim, dtype=np.float32)[one_hot_inds]
    
    one_hot_inds_edge = np.random.randint(low=0, high=4, size=n_total_edges)
    edge_features=np.eye(4, dtype=np.float32)[one_hot_inds_edge]
    #edge_features = np.ones((n_total_edges, edge_feature_dim), dtype=np.float32)
    
    edges = np.array(g.edges(), dtype=np.int32) 
    
    
    from_idx = np.concatenate(from_idx, axis=0)
    to_idx = np.concatenate(to_idx, axis=0)
    
    graph_idx=np.concatenate(graph_idx, axis=0)
    
    node_features = torch.from_numpy(node_features)
    edge_features = torch.from_numpy(edge_features)
    from_idx = torch.from_numpy(from_idx).long()
    to_idx = torch.from_numpy(to_idx).long()
    graph_idx = torch.from_numpy(graph_idx).long()
    
    
    return node_features, edge_features, from_idx, to_idx, graph_idx



def save_model(model, ngpus, save_path, epoch):
    if not os.path.exists(os.path.join(save_path)):
        os.mkdir(os.path.join(save_path))
    
    if ngpus > 1:
        torch.save(model.module.state_dict(), os.path.join(save_path, "GMN_%d.pth" % epoch))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, "GMN_%d.pth" % epoch))       
     

def load_model(model, ngpus, model_path, epoch):
    if ngpus == 1:
        model.load_state_dict(torch.load(os.path.join(model_path, "GMN_%d.pth" % epoch)))
    elif ngpus > 1:
        model.module.load_state_dict(torch.load(os.path.join(model_path, "GMN_%d.pth" % epoch)))
