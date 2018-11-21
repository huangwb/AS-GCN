import tensorflow as tf
import numpy as np

class SparsePlus(object):
    """A wrapper around the tf sparsetensor with more functions implemented"""
    def __init__(self, sparse_tensor):
        self.sparse_tensor = sparse_tensor
        self.rows = self.sparse_tensor.indices[:, 0]
        self.cols = self.sparse_tensor.indices[:, 1]
        self.values = self.sparse_tensor.values
        self.dense_shape = self.sparse_tensor.dense_shape

    def gather_columns(self, needed_col_ids):
        """Gather needed columns from the sparsetensor"""
        support = gather_columns(self.sparse_tensor, needed_col_ids)

        return support

    def gather_rows(self, needed_row_ids):
        """Gather needed rows from the sparsetensor"""
        sp_trans = tf.sparse_transpose(self.sparse_tensor)
        sp_trans_gather = gather_columns(sp_trans, needed_row_ids)
        support = tf.sparse_transpose(sp_trans_gather)

        return support

    def nonzero_columns(self, out_idx=tf.int32):
        """Return non-zero columns"""
        columns, _ = tf.unique(self.cols, out_idx=out_idx)

        return columns


def gather_columns(support, needed_col_ids):
    """Gather needed columns from the sparse support matrix"""
    needed_col_ids = tf.cast(needed_col_ids, tf.int64)
    rows = support.indices[:,0]
    cols = support.indices[:,1]
    values = support.values
    dense_shape = support.dense_shape

    num_samples = tf.size(rows)
    connect = tf.equal(tf.reshape(cols, [-1, 1]), tf.reshape(needed_col_ids, [1, -1]))
    partitions = tf.cast(tf.reduce_any(connect, axis=1), tf.int32)
    samples_to_gather = tf.cast(tf.dynamic_partition(tf.range(num_samples), partitions, 2)[1], tf.int64)

    new_rows = tf.gather(rows, samples_to_gather)
    new_cols = tf.gather(tf.argmax(tf.cast(connect, tf.int64), axis=1), samples_to_gather)
    new_values = tf.gather(values, samples_to_gather)
    dense_shape = tf.stack((dense_shape[0], tf.size(needed_col_ids, out_type=tf.int64)))

    support = tf.SparseTensor(indices=tf.stack((new_rows, new_cols), axis=1), values=new_values, dense_shape=dense_shape)

    return support


def from_adjlist(adj, adj_val):
    """Transfer adj-list format to sparsetensor"""
    u_sampled, index = tf.unique(tf.reshape(adj, [-1]), out_idx=tf.int64)

    row = tf.cast(tf.range(tf.size(index)) // tf.shape(adj)[1], tf.int64)
    col = index
    values = tf.reshape(adj_val, [-1])
    indices = tf.stack([row, col], axis=1)
    dense_shape = tf.cast(tf.stack((tf.shape(adj)[0], tf.size(u_sampled))), tf.int64)

    support = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    return support, u_sampled


def compute_adjlist(sp_adj, max_degree):
    """Transfer sparse adjacent matrix to adj-list format"""
    num_data = sp_adj.shape[0]
    adj = num_data+np.zeros((num_data+1, max_degree), dtype=np.int32)
    adj_val = np.zeros((num_data+1, max_degree), dtype=np.float32)

    for v in range(num_data):
        neighbors = np.nonzero(sp_adj[v, :])[1]
        len_neighbors = len(neighbors)
        if len_neighbors > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
            adj[v] = neighbors
            adj_val[v, :len_neighbors] = sp_adj[v, neighbors].toarray()
        else:
            adj[v, :len_neighbors] = neighbors
            adj_val[v, :len_neighbors] = sp_adj[v, neighbors].toarray()

    return adj, adj_val
