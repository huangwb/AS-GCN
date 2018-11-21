from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.reg_loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)

        # uncomment the following to enable skip connection
        # hidden_12, reg_loss_12 = self.layers[0](self.inputs)
        # hidden_23, reg_loss_23 = self.layers[1](hidden_12)
        # self.outputs = hidden_23
        # hidden_13, reg_loss_13 = self.layers[2](self.inputs)
        # self.outputs += FLAGS.skip*hidden_13
        # self.reg_loss = reg_loss_23
        for layer in self.layers:
            hidden, reg_loss = layer(self.activations[-1])
            self.activations.append(hidden)

        self.reg_loss += reg_loss
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCNAdapt(Model):
    def __init__(self, placeholders, input_dim, sample_train=False, **kwargs):
        super(GCNAdapt, self).__init__(**kwargs)
        self.features = placeholders['features_inputs']
        self.inputs = self.features[0]
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.supports = placeholders['support']
        self.probs = placeholders['prob']
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # with tf.variable_scope(self.name + '_vars'):
        #     self.sample_params = glorot([input_dim, 2], name='sample_params')
        with tf.variable_scope(self.name, reuse=True):
            self.sample_params =tf.get_variable('sample_weights', [input_dim, 2])

        self.support_21 = self._attention(self.supports[0], self.features[0], self.features[1], self.probs[0])
        self.support_32 = self._attention(self.supports[1], self.features[1], self.features[2], self.probs[1])
        # self.attention_31 = tf.sparse_tensor_dense_matmul(self.attention_32, tf.sparse_tensor_to_dense(self.attention_21, validate_indices=False))  # for skip connection

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

        self.loss += FLAGS.var*self.reg_loss

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])

    def _attention(self, support, x_u, x_v, prob):
        sample_params = self.sample_params
        n_v = tf.shape(support)[0]
        n_u = tf.shape(support)[1]
        h_v = tf.matmul(x_v, tf.expand_dims(sample_params[:,0], -1))          # v*1
        h_u = tf.matmul(x_u, tf.expand_dims(sample_params[:,1], -1))          # u*1
        attention = 1/tf.cast(n_u, tf.float32) * (tf.nn.relu(h_v + tf.reshape(h_u, (1, n_u))) + 1)       # v*u
        support = support*(attention/tf.reshape(prob, (1, n_u)))

        return support

    def _build(self):
        self.layers.append(GraphSampleConvolution(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  placeholders=self.placeholders,
                                                  support=self.support_21,
                                                  act=tf.nn.relu,
                                                  dropout=True,
                                                  sparse_inputs=False,
                                                  logging=self.logging))

        self.layers.append(GraphSampleConvolutionReg(input_dim=FLAGS.hidden1,
                                                      output_dim=self.output_dim,
                                                      placeholders=self.placeholders,
                                                      support=self.support_32,
                                                      act=lambda x: x,
                                                      dropout=True,
                                                      logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCNAdaptMix(Model):
    def __init__(self, placeholders, input_dim, sample_train=False, **kwargs):
        super(GCNAdaptMix, self).__init__(**kwargs)
        self.features = placeholders['features_inputs']
        self.inputs = self.features[0]
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.supports = placeholders['support']
        self.probs = placeholders['prob']
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        with tf.variable_scope(self.name, reuse=True):
            self.sample_params =tf.get_variable('sample_weights', [input_dim, 2])

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

        self.loss += FLAGS.var * self.reg_loss

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])

    def _attention(self, support, x_u, x_v, prob):
        sample_params = self.sample_params
        n_v = tf.shape(x_v)[0]
        n_u = tf.shape(x_u)[0]
        h_v = tf.matmul(x_v, tf.expand_dims(sample_params[:,0], -1))          # v*1
        h_u = tf.matmul(x_u, tf.expand_dims(sample_params[:,1], -1))          # u*1
        attention = 1/tf.cast(n_u, tf.float32) * (tf.nn.relu(h_v + tf.reshape(h_u, (1, n_u))) + 1)       # v*u
        support = support*(1*attention/tf.reshape(prob, (1, n_u)))
        return support

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.layers.append(GraphSampleConvolutionReg(input_dim=FLAGS.hidden1,
                                                      output_dim=self.output_dim,
                                                      placeholders=self.placeholders,
                                                      support=self._attention(self.supports[0], self.features[0], self.features[1], self.probs[0]),
                                                      act=lambda x: x,
                                                      dropout=True,
                                                      logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


