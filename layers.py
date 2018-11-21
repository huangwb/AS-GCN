from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                for output in outputs:
                    tf.summary.histogram(self.name + '/outputs', output)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output), 0


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 support=None, sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        if support is None:
            self.support = placeholders['support'][0]
        else:
            self.support = support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(1):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        if not self.featureless:
            pre_sup = dot(x, self.vars['weights_0'],
                          sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_0']
        output = dot(self.support, pre_sup, sparse=True)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output), 0


class GraphSampleConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders,
                 support, #prob, self_features,
                 name = 'con',
                 sparse_support=True,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 dropout=False, featureless=False, **kwargs):
        super(GraphSampleConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.name = name
        self.act = act
        self.support = support
        self.sparse_support = sparse_support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        if not self.featureless:
            pre_sup = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights']

        output = dot(self.support, pre_sup, sparse=self.sparse_support)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output), 0


class GraphSampleConvolutionReg(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders,
                 support, #prob, self_features,
                 name = 'con_reg',
                 sparse_support=True,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 dropout=False, featureless=False, **kwargs):
        super(GraphSampleConvolutionReg, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.name = name
        self.act = act
        self.support = support
        self.sparse_support = sparse_support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        if not self.featureless:
            pre_sup = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights']

        output = dot(self.support, pre_sup, sparse=self.sparse_support)

        n_v = tf.cast(tf.shape(self.support)[0], tf.float32)
        n_u = tf.cast(tf.shape(self.support)[1], tf.float32)

        mean_output = tf.reshape(tf.reduce_mean(output, axis=0), (1, -1))
        if self.sparse_support:
            mean_support = 1.0/tf.cast(n_v, tf.float32)*tf.sparse_reduce_sum(self.support, axis=0)
        else:
            mean_support = 1.0/tf.cast(n_v, tf.float32)*tf.reduce_sum(self.support, axis=0)
        diff = tf.reshape(mean_support, (-1,1))*pre_sup - mean_output
        reg = 1.0 / tf.cast(n_u * self.output_dim, tf.float32) * tf.reduce_sum(diff * diff)


        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output), reg


class GraphSampleConvolutionSkip(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders,
                 support, #prob, self_features,
                 name = 'con_reg',
                 sparse_support=True,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 dropout=False, featureless=False, **kwargs):
        super(GraphSampleConvolutionSkip, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.name = name
        self.act = act
        self.support = support
        self.sparse_support = sparse_support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']


        with tf.variable_scope('con_vars', reuse=True):
            w_12 = tf.get_variable('weights', [input_dim, FLAGS.hidden1])
        with tf.variable_scope('con_reg_vars', reuse=True):
            w_23 = tf.get_variable('weights', [FLAGS.hidden1, output_dim])
        self.vars['weights'] = tf.matmul(w_12, w_23)

        if self.bias:
            self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        if not self.featureless:
            pre_sup = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights']

        output = dot(self.support, pre_sup, sparse=self.sparse_support)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output), 0



