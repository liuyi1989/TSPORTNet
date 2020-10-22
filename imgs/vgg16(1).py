import os
import sys

import numpy as np
import tensorflow as tf

import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope

VGG_MEAN = [103.939, 116.779, 123.68]

# https://github.com/machrisaa/tensorflow-vgg

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "vgg16.npy")
            print(path)
            vgg16_npy_path = path

        self.data_dict = np.load(vgg16_npy_path,encoding="latin1").item()
        print("npy file loaded")

    def build(self, input, train=False):

        self.conv1_1 = self._conv_layer(input, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5')


    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            #tf.add_to_collection('conv_out', relu)
            return relu

    def _fc_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):

        #W_regul = lambda x: self.L2(x)

        #return tf.get_variable(name="filter",
        #                       initializer=self.data_dict[name][0],
        #                       trainable=True,
        #                       regularizer=W_regul)
        return tf.Variable(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights")

    def L2(self, tensor, wd=0.001):
        return tf.mul(tf.nn.l2_loss(tensor), wd, name='L2-Loss')
    
    
class ResNet:
    def __init__(self, ResNet_npy_path=None):
        if ResNet_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "ResNet-pretrained-model/ResNet152.npy")
            print(path)
            ResNet_npy_path = path

        self.data_dict = np.load(ResNet_npy_path,encoding="latin1").item()
        print("npy file loaded")
        
    def build(self, input, train=False):
        MODEL_DEPTH = 50
        config_map = {
          50:  [3,4,6,3],
          101: [3,4,23,3],
          152: [3,8,36,3]
        }
        config = config_map[MODEL_DEPTH]
        
        weight_decay = 1e-4
        init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
        bn_params = {
          # Decay for the moving averages.
          'decay': 0.9,
          'center': True,
          'scale': True,
          # epsilon to prevent 0s in variance.
          'epsilon': 1e-5,
          # None to force the updates
          'updates_collections': None,
          'is_training': True,
        }        
        
        #input = normalize_input(input)
        #input = tf.pad(input, [[0,0],[3,3],[3,3],[0,0]])
        #net = layers.convolution2d(input, 64, 7, stride=2, padding='VALID',
        self.net_conv0 = layers.convolution2d(input, 64, 7, stride=2, padding='SAME',
            activation_fn=None, weights_initializer=init_func,
            normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
            weights_regularizer=layers.l2_regularizer(weight_decay), scope='conv0')
        
        self.net_pool0 = layers.max_pool2d(self.net_conv0, 3, stride=2, padding='SAME', scope='pool0')
        self.net_group0 = self.layer(self.net_pool0, 'group0', 64, config[0], 1)
        self.net_group1 = self.layer(self.net_group0, 'group1', 128, config[1], 2)
        self.net_group2 = self.layer(self.net_group1, 'group2', 256, config[2], 2)
        self.net_group3 = self.layer(self.net_group2, 'group3', 512, config[3], 2)
        #self.net_fully = tf.nn.relu(self.net_group3)
        #in_size = net.get_shape().as_list()[1:3]
        #self.net_GAP = layers.avg_pool2d(self.net_fully, kernel_size=in_size, scope='global_avg_pool')
        #self.net_flatten = layers.flatten(self.net_GAP, scope='flatten')
        #logits = layers.fully_connected(self.net_flatten, 1000, activation_fn=None, scope='fc1000')
        #return logits  
    
        #for k, v in self.data_dict.items():
            #print(v)
            #self.newname = self.name_conversion(k)
            #self.resnet_param[self.newname] = v
            #print(v)        
    #self.resnet_param = {}
    
    def layer(self, net, name, num_maps, num_layers, stride):
        with tf.variable_scope(name):
            for i in range(num_layers):
                with tf.variable_scope('block{}'.format(i)):
                    s = stride if i == 0 else 1
                    net = self.bottleneck(net, num_maps, s)
            return net  
        
    def bottleneck(self, net, num_maps, stride):
        weight_decay = 1e-4
        init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
        bn_params = {
          # Decay for the moving averages.
          'decay': 0.9,
          'center': True,
          'scale': True,
          # epsilon to prevent 0s in variance.
          'epsilon': 1e-5,
          # None to force the updates
          'updates_collections': None,
          'is_training': True,
        }                
        net = tf.nn.relu(net)
        bottom_net = net
        with arg_scope([layers.convolution2d], 
                       padding='SAME', activation_fn=tf.nn.relu,
                       normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
                       weights_initializer=init_func,
                       weights_regularizer=layers.l2_regularizer(weight_decay)):
            
            net = layers.convolution2d(net, num_maps, kernel_size=1, stride=stride, scope='conv1')
            net = layers.convolution2d(net, num_maps, kernel_size=3, scope='conv2')
            net = layers.convolution2d(net, num_maps * 4, kernel_size=1,
                                       activation_fn=None, scope='conv3')
            return net + self.shortcut(bottom_net, num_maps * 4, stride)
        
    def shortcut(self, net, num_maps_out, stride):
        num_maps_in = net.get_shape().as_list()[-1]
        if num_maps_in != num_maps_out:
            return layers.convolution2d(net, num_maps_out, kernel_size=1, stride=stride,
                                        activation_fn=None, scope='convshortcut')
        return net
        
        
    def name_conversion(self, caffe_layer_name):
        NAME_MAP = {
            'bn_conv1/beta': 'conv0/BatchNorm/beta:0',
            'bn_conv1/gamma': 'conv0/BatchNorm/gamma:0',
            'bn_conv1/mean/EMA': 'conv0/BatchNorm/moving_mean:0',
            'bn_conv1/variance/EMA': 'conv0/BatchNorm/moving_variance:0',
            'conv1/W': 'conv0/weights:0', 'conv1/b': 'conv0/biases:0',
            'fc1000/W': 'fc1000/weights:0', 'fc1000/b': 'fc1000/biases:0'}
        if caffe_layer_name in NAME_MAP:
            return NAME_MAP[caffe_layer_name]
    
        s = re.search('([a-z]+)([0-9]+)([a-z]+)_', caffe_layer_name)
        if s is None:
            s = re.search('([a-z]+)([0-9]+)([a-z]+)([0-9]+)_', caffe_layer_name)
            layer_block_part1 = s.group(3)
            layer_block_part2 = s.group(4)
            assert layer_block_part1 in ['a', 'b']
            layer_block = 0 if layer_block_part1 == 'a' else int(layer_block_part2)
        else:
            layer_block = ord(s.group(3)) - ord('a')
        layer_type = s.group(1)
        layer_group = s.group(2)
    
        layer_branch = int(re.search('_branch([0-9])', caffe_layer_name).group(1))
        assert layer_branch in [1, 2]
        if layer_branch == 2:
            layer_id = re.search('_branch[0-9]([a-z])/', caffe_layer_name).group(1)
            layer_id = ord(layer_id) - ord('a') + 1
    
        type_dict = {'res':'conv', 'bn':'BatchNorm'}
        name_map = {'/W': '/weights:0', '/b': '/biases:0', '/beta': '/beta:0',
                    '/gamma': '/gamma:0', '/mean/EMA': '/moving_mean:0',
                    '/variance/EMA': '/moving_variance:0'}
    
        tf_name = caffe_layer_name[caffe_layer_name.index('/'):]
        if tf_name in name_map:
            tf_name = name_map[tf_name]
        if layer_type == 'res':
            layer_type = type_dict[layer_type] + \
            (str(layer_id) if layer_branch == 2 else 'shortcut')
        elif layer_branch == 2:
            layer_type = 'conv' + str(layer_id) + '/' + type_dict[layer_type]
        elif layer_branch == 1:
            layer_type = 'convshortcut/' + type_dict[layer_type]
        tf_name = 'group{}/block{}/{}'.format(int(layer_group) - 2,
                                            layer_block, layer_type) + tf_name
        return tf_name
        
              
          


