from utils import *

def transferRedditDataFormat(dataset_dir, dataset):
    G = json_graph.node_link_graph(json.load(open(dataset_dir + dataset + "-G.json")))
    labels = json.load(open(dataset_dir + dataset + "-class_map.json"))

    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n
    if isinstance(list(labels.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    labels = {conversion(k):lab_conversion(v) for k,v in labels.items()}


    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n]['test']]
    val_ids = [n for n in G.nodes() if G.node[n]['val']]
    train_labels = [labels[i] for i in train_ids]
    print(len(train_labels[0]))
    test_labels = [labels[i] for i in test_ids]
    val_labels = [labels[i] for i in val_ids]
    feats = np.load(dataset_dir + dataset + "-feats.npy")
    ## Logistic gets thrown off by big counts, so log transform num comments and score
    feats[:, 0] = np.log(feats[:, 0] + 1.0)
    feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
    feat_id_map = json.load(open(dataset_dir + dataset + "-id_map.json"))
    feat_id_map = {conversion(id):int(val) for id, val in feat_id_map.iteritems()}
    #feat_id_map = {conversion(k):int(v) for k,v in feat_id_map.items()}
    # train_feats = feats[[feat_id_map[id] for id in train_ids]]
    # test_feats = feats[[feat_id_map[id] for id in test_ids]]

    numNode = len(feat_id_map)
    adj = sp.lil_matrix((numNode,numNode))
    #adj = sp.csr_matrix((numNode,numNode))
    for edge in G.edges():
        adj[feat_id_map[edge[0]], feat_id_map[edge[1]]] = 1
    sp.save_npz(dataset_dir+dataset+"_adj", adj.tocsr())

    train_index = [feat_id_map[id] for id in train_ids]
    val_index = [feat_id_map[id] for id in val_ids]
    test_index = [feat_id_map[id] for id in test_ids]
    np.savez(dataset_dir+dataset, feats = feats, y_train=train_labels, y_val=val_labels, y_test = test_labels, train_index = train_index,
            val_index=val_index, test_index = test_index)


if __name__=="__main__":

    data_dir = './data/'
    dataset= 'reddit'  # 'ppi'

    transferRedditDataFormat(data_dir, dataset)
