import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.layers.python.layers import batch_norm, variance_scaling_initializer


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def elu(x, name="elu"):
    return tf.nn.elu(x)

def conv2d(input_, output_dim,k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,name="conv2d",Reuse=False):
    with tf.variable_scope(name,reuse=Reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        #conv=tf.nn.bias_add(conv, biases)
    return conv

def dilated_conv(input_,output_dim,rate,kernel_h=3, kernel_w=3,stddev=0.02,name="dilated_conv",Reuse=False):
    with tf.variable_scope(name,reuse=Reuse):
        w=tf.get_variable('w', [kernel_h,kernel_w,input_.get_shape()[-1], output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        dil_conv=tf.nn.atrous_conv2d(input_,w, rate=rate, padding="SAME")
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        dil_conv=tf.reshape(tf.nn.bias_add(dil_conv, biases), dil_conv.get_shape())

    return dil_conv

def dilated_conv_with_drop(input_,output_dim,rate,kernel_h=3, kernel_w=3,stddev=0.02,drop_prob=0.5,TRAIN=True, name="dilated_conv_drop",Reuse=False):
    with tf.variable_scope(name,reuse=Reuse):
        w=tf.get_variable('w', [kernel_h,kernel_w,input_.get_shape()[-1], output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        dil_conv=tf.nn.atrous_conv2d(input_,w, rate=rate, padding="SAME")
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        dil_conv = elu(tf.nn.bias_add(dil_conv, biases))
        spatial_dropout=tf.keras.layers.SpatialDropout2D(rate=drop_prob,data_format='channels_last')
        dil_conv=spatial_dropout.call(dil_conv,training=TRAIN)

    return dil_conv



def maxpool_layer(x, size, stride, name):
    with tf.name_scope(name):
        x = tf.layers.max_pooling2d(x, size, stride, padding='SAME')#pooling size

    return x


def unpool_with_argmax(pool, ind, name = None, ksize=[1, 2, 2, 1]):

    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(name):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret


def conv2d_res(x, n_in, n_out, k, s, p='SAME', bias=False, scope='conv',Reuse=False):
  with tf.variable_scope(scope,reuse=Reuse):
    kernel = tf.Variable(
      tf.truncated_normal([k, k, n_in, n_out],
        stddev=math.sqrt(2/(k*k*n_in))),
      name='weight')
    tf.add_to_collection('weights', kernel)
    conv = tf.nn.conv2d(x, kernel, [1,s,s,1], padding=p)
    if bias:
      bias = tf.get_variable('bias', [n_out], initializer=tf.constant_initializer(0.0))
      tf.add_to_collection('biases', bias)
      conv = tf.nn.bias_add(conv, bias)
  return conv

def batch_norm_res(x, n_out, phase_train, scope='bn', affine=True):
  """
  Batch normalization on convolutional maps.
  Args:
    x: Tensor, 4D BHWD input maps
    n_out: integer, depth of input maps
    phase_train: boolean tf.Variable, true indicates training phase
    scope: string, variable scope
    affine: whether to affine-transform outputs
  Return:
    normed: batch-normalized maps
  """
  with tf.variable_scope(scope):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
      name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
      name='gamma', trainable=True)
    tf.add_to_collection('biases', beta)
    tf.add_to_collection('weights', gamma)

    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')#calaulate the mean and variance of x, global normalization과 같이 사용하기 위해서는 axis=[0,1,2]로 주어야 한다.
    ema = tf.train.ExponentialMovingAverage(decay=0.99)#maintains moving averages of variables by employin an exponential decay.

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])#여기서 질실적인 연산이 이루어진다고 생각하면 된다. 내 생각에는 avg=pre_avg*decay+(1-decay)*current_variable인듯 하다.
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)#return a tensor with the same shape and contents as input

    mean, var = control_flow_ops.cond(phase_train,
      mean_var_with_update,
      lambda: (ema.average(batch_mean), ema.average(batch_var)))#phase_train의 값을 보고 true일 경우에는 mean_var_with_update를 return하고 false일 경우에는 아래의 lambda함수를 return한다.

    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,beta, gamma, 1e-3, affine)
    #여기서 beta, gamma를 학습한다고 보면 된다.
    #affine이 있는 자리가 sale_after_normalization
    #1e-3 : A small float number to avoid dividing by 0
    #affine=A bool indicating whether the resulted tensor needs to be multiplied with gamma.
  return normed


def residual_group(x, n_in, n_out, residual_net_n, first_subsample, phase_train, scope='res_group'):
  with tf.variable_scope(scope):
    y = residual_block(x, n_in, n_out, first_subsample, phase_train, scope='block_1')
    for i in range(residual_net_n - 1):
      y = residual_block(y, n_out, n_out, False, phase_train, scope='block_%d' % (i + 2))
  return y


def residual_block(x, n_in, n_out, subsample, phase_train, scope='res_block'):
  with tf.variable_scope(scope):
    if subsample:#subsample means that stride is 2, so the W,H is decrease.
      y = conv2d_res(x, n_in, n_out, 3, 2, 'SAME', False, scope='conv_1')#here, false means that no bias.
      shortcut = conv2d_res(x, n_in, n_out, 3, 2, 'SAME',False, scope='shortcut')#If we subsample, we need shortcut.
    else:
      y = conv2d_res(x, n_in, n_out, 3, 1, 'SAME', False, scope='conv_1')
      shortcut = tf.identity(x, name='shortcut')#if the dimension is same, we just need identity value.
    y = batch_norm_res(y, n_out, phase_train, scope='bn_1')
    y = tf.nn.relu(y, name='relu_1')
    y = conv2d_res(y, n_out, n_out, 3, 1, 'SAME', True, scope='conv_2')
    y = batch_norm_res(y, n_out, phase_train, scope='bn_2')
    y = y + shortcut
    y = tf.nn.relu(y, name='relu_2')
  return y



def Deconv2DfromVoxelNet(Cin, Cout, k, s, p, input, training=True, name='deconv'):
    temp_p = np.array(p)#p is always (0,0)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
    pad = tf.pad(input, paddings, "CONSTANT")
    with tf.variable_scope(name) as scope:
        temp_conv = tf.layers.conv2d_transpose(
            pad, Cout, k, strides=s, padding="SAME", reuse=tf.AUTO_REUSE, name=scope)#stride is actually factor for enlarge, for example s is (2,2), the H, W -> 2W, 2H
        temp_conv = tf.layers.batch_normalization(
            temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
        return tf.nn.relu(temp_conv)


def Deconv2DfromVoxelNet_2(Cin, Cout, k, s, p, input, training=True, name='deconv'):
    temp_p = np.array(p)#p is always (0,0)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
    pad = tf.pad(input, paddings, "CONSTANT")
    with tf.variable_scope(name) as scope:
        temp_conv = tf.layers.conv2d_transpose(
            pad, Cout, k, strides=s, padding="SAME", reuse=tf.AUTO_REUSE, name=scope)#stride is actually factor for enlarge, for example s is (2,2), the H, W -> 2W, 2H
        temp_conv = batch_norm_res(temp_conv,Cout,training,scope='voxelDeconv')
        return tf.nn.relu(temp_conv)

