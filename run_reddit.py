from __future__ import division
from __future__ import print_function

import time
import os
from inits import *
from sampler import *
from models import GCNAdapt, GCNAdaptMix

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'reddit', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_adapt_mix', 'Model string.')  # 'gcn', 'gcn_appr'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 64, 'Maximum Chebyshev for constructing the adjacent matrix.')
flags.DEFINE_integer('gpu', '0', 'The gpu to be applied.')
flags.DEFINE_string('sampler_device', 'cpu', 'The device for sampling: cpu or gpu.')
flags.DEFINE_integer('rank', 512, 'The number of nodes per layer.')
flags.DEFINE_integer('skip', 0, 'If use skip connection.')
flags.DEFINE_float('var', 0.5, 'If use variance reduction.')
# Load data
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)


def main(rank1, rank0):

    # Prepare data
    adj, adj_train, adj_val_train, features, train_features, y_train, y_test, test_index = prepare_reddit(FLAGS.max_degree)
    print('preparation done!')

    max_degree = FLAGS.max_degree
    num_train = adj_train.shape[0]-1
    input_dim = features.shape[1]
    scope = 'test'

    if FLAGS.model == 'gcn_adapt_mix':
        num_supports = 1
        propagator = GCNAdaptMix
        test_supports = [sparse_to_tuple(adj[test_index, :])]
        test_features = [features, features[test_index, :]]
        test_probs = [np.ones(adj.shape[0])]
        layer_sizes = [rank1, 256]
    elif FLAGS.model == 'gcn_adapt':
        num_supports = 2
        propagator = GCNAdapt
        test_supports = [sparse_to_tuple(adj), sparse_to_tuple(adj[test_index, :])]
        test_features = [features, features, features[test_index, :]]
        test_probs = [np.ones(adj.shape[0]), np.ones(adj.shape[0])]
        layer_sizes = [rank0, rank1, 256]
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'batch': tf.placeholder(tf.int32),
        'adj': tf.placeholder(tf.int32, shape=(num_train+1, max_degree)),
        'adj_val': tf.placeholder(tf.float32, shape=(num_train+1, max_degree)),
        'features': tf.placeholder(tf.float32, shape=train_features.shape),
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'prob': [tf.placeholder(tf.float32) for _ in range(num_supports)],
        'features_inputs': [tf.placeholder(tf.float32, shape=(None, input_dim)) for _ in range(num_supports+1)],
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Sampling parameters shared by the sampler and model
    with tf.variable_scope(scope):
        w_s = glorot([features.shape[-1], 2], name='sample_weights')

    # Create sampler
    if FLAGS.sampler_device == 'cpu':
        with tf.device('/cpu:0'):
            sampler_tf = SamplerAdapt(placeholders, input_dim=input_dim, layer_sizes=layer_sizes, scope=scope)
            features_sampled, support_sampled, p_u_sampled = sampler_tf.sampling(placeholders['batch'])
    else:
        sampler_tf = SamplerAdapt(placeholders, input_dim=input_dim, layer_sizes=layer_sizes, scope=scope)
        features_sampled, support_sampled, p_u_sampled = sampler_tf.sampling(placeholders['batch'])

    # Create model
    model = propagator(placeholders, input_dim=input_dim, logging=True, name=scope)

    # Initialize session
    config = tf.ConfigProto(device_count={"CPU": 1},
                            inter_op_parallelism_threads=0,
                            intra_op_parallelism_threads=0,
                            log_device_placement=False)
    sess = tf.Session(config=config)

    # Define model evaluation function
    def evaluate(features, support, prob_norm, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict_with_prob(features, support, prob_norm, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    def increment_evaluate(features, ADJ, index, labels, mask, placeholders):
        t_test = time.time()
        num_nodes = len(index)
        loss = 0
        acc = 0
        batchsize = 2048
        batch_end = 0
        for batch_start in xrange(0, num_nodes, batchsize):
            batch_end = batch_start + batchsize
            index_batch = index[batch_start:batch_end]
            label_batch = labels[batch_start:batch_end]
            feed_dict_val = construct_feed_dict_with_prob([features, features[index_batch, :]],
                                                          [sparse_to_tuple(ADJ[index_batch, :])],
                                                          [np.ones(ADJ.shape[0])],
                                                          label_batch, mask, placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            loss += outs_val[0]*batchsize
            acc += outs_val[1]*batchsize

        if batch_end < num_nodes:
            index_batch = index[batch_end:]
            label_batch = labels[batch_end:]
            feed_dict_val = construct_feed_dict_with_prob([features, features[index_batch, :]],
                                                          [sparse_to_tuple(ADJ[index_batch, :])],
                                                          [np.ones(ADJ.shape[0])],
                                                          label_batch, mask, placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            loss += outs_val[0]*(num_nodes-batch_end)
            acc += outs_val[1]*(num_nodes-batch_end)

        return loss/num_nodes, acc/num_nodes, (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={placeholders['adj']: adj_train,
                                                           placeholders['adj_val']: adj_val_train,
                                                           placeholders['features']: train_features})

    # Prepare training
    saver = tf.train.Saver()
    save_dir = "tmp/" + FLAGS.dataset + '_' + str(FLAGS.skip) + '_' + str(FLAGS.var) + '_' + str(FLAGS.gpu)
    acc_val = []
    acc_train = []
    train_time = []
    train_time_sample = []
    max_acc = 0
    t = time.time()
    # Train model
    for epoch in range(FLAGS.epochs):

        sample_time = 0
        t1 = time.time()

        for batch in iterate_minibatches_listinputs([y_train, np.arange(num_train)], batchsize=256, shuffle=True):
            [y_train_batch, train_batch] = batch

            if sum(train_batch) < 1:
                continue
            ts = time.time()
            features_inputs, supports, probs = sess.run([features_sampled, support_sampled, p_u_sampled],
                                                        feed_dict={placeholders['batch']:train_batch})
            sample_time += time.time()-ts

            # Construct feed dictionary
            feed_dict = construct_feed_dict_with_prob(features_inputs, supports, probs, y_train_batch, [],
                                                      placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy, model.sample_params], feed_dict=feed_dict)
            acc_train.append(outs[-2])

        train_time_sample.append(time.time()-t1)
        train_time.append(time.time()-t1-sample_time)
        # Validation
        cost, acc, duration = increment_evaluate(features, adj, test_index, y_test, [], placeholders)
        acc_val.append(acc)
        # if epoch > 50 and acc>max_acc:
        #     max_acc = acc
        #     saver.save(sess, save_dir + ".ckpt")

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(train_time_sample[epoch]))

    train_duration = np.mean(np.array(train_time_sample))
    # Testing
    # if os.path.exists(save_dir + ".ckpt.index"):
    #     saver.restore(sess, save_dir + ".ckpt")
    #     print('Loaded the  best ckpt.')
    # test_cost, test_acc, test_duration = increment_evaluate(features, adj, test_index, y_test, [], placeholders)
    # print("rank1 = {}".format(rank1), "rank0 = {}".format(rank0), "cost=", "{:.5f}".format(test_cost),
    #       "accuracy=", "{:.5f}".format(test_acc), "training time per epoch=", "{:.5f}".format(train_duration))


if __name__ == "__main__":

    print("DATASET:", FLAGS.dataset)
    main(FLAGS.rank,FLAGS.rank)
