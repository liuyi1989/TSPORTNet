import tensorflow as tf
import vgg16
import cv2
import numpy as np

import multiprocessing
import os

import tensorflow.contrib.slim as slim
from config import cfg

#import sys
#import importlib
#importlib.reload(sys)

img_size = 352
label_size = img_size
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Model:
    def __init__(self):
        self.vgg = vgg16.Vgg16()

        self.input_holder = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
        self.label_holder = tf.placeholder(tf.float32, [label_size*label_size, 2])

        self.sobel_fx, self.sobel_fy = self.sobel_filter()

        self.contour_th = 1.5
        self.contour_weight = 0.0001

    def build_model(self):

        #build the VGG-16 model
        vgg = self.vgg
        vgg.build(self.input_holder)

        fea_dim = 128
        data_size = 11
        weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)
        
        
        # Local Score
        self.Fea_P1 = self.dilation(vgg.conv1_2, 64, fea_dim/4, 'Fea_P1')
        self.Fea_P2 = self.dilation(vgg.conv2_2, 128, fea_dim/4, 'Fea_P2')
        self.Fea_P3 = self.dilation(vgg.conv3_3, 256, fea_dim/4, 'Fea_P3')
        self.Fea_P4 = self.dilation(vgg.conv4_3, 512, fea_dim/4, 'Fea_P4')
        self.Fea_P5 = self.dilation(vgg.conv5_3, 512, fea_dim/4, 'Fea_P5')
        
        with tf.variable_scope('relu_conv1') as scope:
            self.output = slim.conv2d(vgg.conv5_3, num_outputs=cfg.B, kernel_size=[
                                 5, 5], stride=2, padding='SAME', scope=scope, activation_fn=tf.nn.relu)
            #data_size = int(np.floor((data_size - 4) / 2))
 
        with tf.variable_scope('primary_caps1') as scope:
            pose = slim.conv2d(self.output, num_outputs=cfg.B * 16,
                               kernel_size=[1, 1], stride=1, padding='VALID', scope=scope, activation_fn=None)
            activation = slim.conv2d(self.output, num_outputs=cfg.B, kernel_size=[
                                     1, 1], stride=1, padding='VALID', scope='primary_caps/activation', activation_fn=tf.nn.sigmoid)
            pose = tf.reshape(pose, shape=[cfg.batch_size, data_size, data_size, cfg.B, 16])
            activation = tf.reshape(
                activation, shape=[cfg.batch_size, data_size, data_size, cfg.B, 1])
            self.output = tf.concat([pose, activation], axis=4)
            self.output = tf.reshape(self.output, shape=[cfg.batch_size, data_size, data_size, -1])
            
        with tf.variable_scope('conv_caps1') as scope:
            self.output = self.kernel_tile(self.output, 1, 1)
            #data_size = int(np.floor((data_size - 2) / 2))
            #data_size = int(np.floor(data_size / 2))
            self.output = tf.reshape(self.output, shape=[cfg.batch_size *
                                               data_size * data_size, 1 * 1 * cfg.B, 17])
            activation = tf.reshape(self.output[:, :, 16], shape=[
                                    cfg.batch_size * data_size * data_size, 1 * 1 * cfg.B, 1])

            with tf.variable_scope('v') as scope:
                votes = self.mat_transform(self.output[:, :, :16], cfg.C, weights_regularizer, tag=True)

            with tf.variable_scope('routing') as scope:
                miu, activation, _ = self.em_routing(votes, activation, cfg.C, weights_regularizer)

            pose = tf.reshape(miu, shape=[cfg.batch_size, data_size, data_size, cfg.C, 16])
            activation = tf.reshape(
                activation, shape=[cfg.batch_size, data_size, data_size, cfg.C, 1])         
            self.output = tf.reshape(tf.concat([pose, activation], axis=4), [
                                cfg.batch_size, data_size, data_size, -1])
            
            
        with tf.variable_scope('deconv_caps1') as scope:
            data_size = int(np.floor(data_size * 2))
            self.output = tf.image.resize_images(self.output, [data_size, data_size]) 
            self.output = self.kernel_tile(self.output, 1, 1)
                      
            
            self.output = tf.reshape(self.output, shape=[cfg.batch_size *
                                               data_size * data_size, 1 * 1 * cfg.C, 17])
            activation = tf.reshape(self.output[:, :, 16], shape=[
                                    cfg.batch_size * data_size * data_size, 1 * 1 * cfg.C, 1])

            with tf.variable_scope('v') as scope:
                votes = self.mat_transform(self.output[:, :, :16], cfg.D, weights_regularizer)

            with tf.variable_scope('routing') as scope:
                miu, activation, _ = self.em_routing(votes, activation, cfg.D, weights_regularizer)

            #pose = tf.reshape(miu, shape=[cfg.batch_size * data_size * data_size, cfg.D, 16])
            #activation = tf.reshape(
                #activation, shape=[cfg.batch_size * data_size * data_size, cfg.D, 1])
            pose = tf.reshape(miu, shape=[cfg.batch_size, data_size, data_size, cfg.D, 16])
            activation = tf.reshape(
                activation, shape=[cfg.batch_size, data_size , data_size, cfg.D, 1])       
            
        self.output1 = tf.reshape(activation, shape=[cfg.batch_size, data_size, data_size, cfg.D])              
            
            
        #self.caps4 = tf.image.resize_images(self.output, [44, 44])    
        #self.output_Up4 = tf.nn.relu(self.Deconv_2d(self.output, [1, 44, 44, fea_dim], 5, 2, name='output_Deconv4'))
        self.dec5 = self.Conv_2d(tf.concat([self.Fea_P5, self.output1], axis=3), [1, 1, fea_dim + cfg.D, fea_dim], 0.01, padding='VALID', name='Fea_P5_Concat')
        #self.dec4 = self.Conv_2d(self.output_Up4, [1, 1, fea_dim, fea_dim], 0.01, padding='VALID', name='dec4')
        
        #decoder1
        self.caps4 = tf.image.resize_images(self.output1, [44, 44])
        self.dec4 = self.OGU1(self.caps4, self.dec5, self.Fea_P4, fea_dim, 44, cfg.D, 1, 4)
        
        
        
        self.caps3 = tf.image.resize_images(self.caps4, [88, 88])
        self.dec3 = self.OGU1(self.caps3, self.dec4, self.Fea_P3, fea_dim, 88, cfg.D, 1, 3) 
        
        self.caps2 = tf.image.resize_images(self.caps3, [176, 176])
        self.dec2 = self.OGU1(self.caps2, self.dec3, self.Fea_P2, fea_dim, 176, cfg.D, 1, 2)
        
        self.caps1 = tf.image.resize_images(self.caps2, [352, 352])
        self.dec1 = self.OGU1(self.caps1, self.dec2, self.Fea_P1, fea_dim, 352, cfg.D, 1, 1)
        
        self.Fea_Pdec1 = self.dilation(self.dec1, fea_dim, fea_dim/4, 'Fea_Pdec1')
        self.Fea_Pdec2 = self.dilation(self.dec2, fea_dim, fea_dim/4, 'Fea_Pdec2')
        self.Fea_Pdec3 = self.dilation(self.dec3, fea_dim, fea_dim/4, 'Fea_Pdec3')
        self.Fea_Pdec4 = self.dilation(self.dec4, fea_dim, fea_dim/4, 'Fea_Pdec4')
        self.Fea_Pdec5 = self.dilation(self.dec5, fea_dim, fea_dim/4, 'Fea_Pdec5')
        
        #encoder2
        self.pool2_1, self.enc2_1, self.output_Score1_1, self.output_Score1_1_S = self.SWG1(self.dec1, self.Fea_Pdec1, fea_dim, fea_dim, 64, 2, 1)
        
        self.pool2_2, self.enc2_2, self.output_Score1_2_S = self.SWG2(self.pool2_1, self.Fea_Pdec2, self.output_Score1_1_S[0], 176, 64, fea_dim, 128, 2, 2)
        
        self.pool2_3, self.enc2_3, self.output_Score1_3_S = self.SWG2(self.pool2_2, self.Fea_Pdec3, self.output_Score1_2_S, 88, 128, fea_dim, 256, 2, 3)
        
        self.pool2_4, self.enc2_4, self.output_Score1_4_S = self.SWG2(self.pool2_3, self.Fea_Pdec4, self.output_Score1_3_S, 44, 256, fea_dim, 512, 2, 4)
        
        self.pool2_5, self.enc2_5, self.output_Score1_5_S = self.SWG2(self.pool2_4, self.Fea_Pdec5, self.output_Score1_4_S, 22, 512, fea_dim, 512, 2, 5)

        self.Fea_P2_1 = self.dilation(self.enc2_1, 64, fea_dim/4, 'Fea_P2_1')
        self.Fea_P2_2 = self.dilation(self.enc2_2, 128, fea_dim/4, 'Fea_P2_2')
        self.Fea_P2_3 = self.dilation(self.enc2_3, 256, fea_dim/4, 'Fea_P2_3')
        self.Fea_P2_4 = self.dilation(self.enc2_4, 512, fea_dim/4, 'Fea_P2_4')
        self.Fea_P2_5 = self.dilation(self.enc2_5, 512, fea_dim/4, 'Fea_P2_5')    
        
        data_size = int(np.floor(data_size / 2))
        
        with tf.variable_scope('relu_conv2') as scope:
            self.output = slim.conv2d(self.enc2_5, num_outputs=cfg.B, kernel_size=[
                                 5, 5], stride=2, padding='SAME', scope=scope, activation_fn=tf.nn.relu)
            #data_size = int(np.floor((data_size - 4) / 2))
 
        with tf.variable_scope('primary_caps2') as scope:
            pose = slim.conv2d(self.output, num_outputs=cfg.B * 16,
                               kernel_size=[1, 1], stride=1, padding='VALID', scope=scope, activation_fn=None)
            activation = slim.conv2d(self.output, num_outputs=cfg.B, kernel_size=[
                                     1, 1], stride=1, padding='VALID', scope='primary_caps/activation', activation_fn=tf.nn.sigmoid)
            pose = tf.reshape(pose, shape=[cfg.batch_size, data_size, data_size, cfg.B, 16])
            activation = tf.reshape(
                activation, shape=[cfg.batch_size, data_size, data_size, cfg.B, 1])
            self.output = tf.concat([pose, activation], axis=4)
            self.output = tf.reshape(self.output, shape=[cfg.batch_size, data_size, data_size, -1])
            
        with tf.variable_scope('conv_caps2') as scope:
            self.output = self.kernel_tile(self.output, 1, 1)
            #data_size = int(np.floor((data_size - 2) / 2))
            #data_size = int(np.floor(data_size / 2))
            self.output = tf.reshape(self.output, shape=[cfg.batch_size *
                                               data_size * data_size, 1 * 1 * cfg.B, 17])
            activation = tf.reshape(self.output[:, :, 16], shape=[
                                    cfg.batch_size * data_size * data_size, 1 * 1 * cfg.B, 1])

            with tf.variable_scope('v') as scope:
                votes = self.mat_transform(self.output[:, :, :16], cfg.C, weights_regularizer, tag=True)

            with tf.variable_scope('routing') as scope:
                miu, activation, _ = self.em_routing(votes, activation, cfg.C, weights_regularizer)

            pose = tf.reshape(miu, shape=[cfg.batch_size, data_size, data_size, cfg.C, 16])
            activation = tf.reshape(
                activation, shape=[cfg.batch_size, data_size, data_size, cfg.C, 1])         
            self.output = tf.reshape(tf.concat([pose, activation], axis=4), [
                                cfg.batch_size, data_size, data_size, -1])
            
            
        with tf.variable_scope('deconv_caps2') as scope:
            data_size = int(np.floor(data_size * 2))
            self.output = tf.image.resize_images(self.output, [data_size, data_size]) 
            self.output = self.kernel_tile(self.output, 1, 1)
                      
            
            self.output = tf.reshape(self.output, shape=[cfg.batch_size *
                                               data_size * data_size, 1 * 1 * cfg.C, 17])
            activation = tf.reshape(self.output[:, :, 16], shape=[
                                    cfg.batch_size * data_size * data_size, 1 * 1 * cfg.C, 1])

            with tf.variable_scope('v') as scope:
                votes = self.mat_transform(self.output[:, :, :16], cfg.D, weights_regularizer)

            with tf.variable_scope('routing') as scope:
                miu, activation, _ = self.em_routing(votes, activation, cfg.D, weights_regularizer)

            #pose = tf.reshape(miu, shape=[cfg.batch_size * data_size * data_size, cfg.D, 16])
            #activation = tf.reshape(
                #activation, shape=[cfg.batch_size * data_size * data_size, cfg.D, 1])
            pose = tf.reshape(miu, shape=[cfg.batch_size, data_size, data_size, cfg.D, 16])
            activation = tf.reshape(
                activation, shape=[cfg.batch_size, data_size , data_size, cfg.D, 1])       
            
        self.output2 = tf.reshape(activation, shape=[cfg.batch_size, data_size, data_size, cfg.D])                          
            
        #self.caps4 = tf.image.resize_images(self.output, [44, 44])    
        #self.output_Up4 = tf.nn.relu(self.Deconv_2d(self.output, [1, 44, 44, fea_dim], 5, 2, name='output_Deconv4'))
        self.dec2_5 = self.Conv_2d(tf.concat([self.Fea_P2_5, self.output2], axis=3), [1, 1, fea_dim + cfg.D, fea_dim], 0.01, padding='VALID', name='Fea_P2_5_Concat')
        #self.dec4 = self.Conv_2d(self.output_Up4, [1, 1, fea_dim, fea_dim], 0.01, padding='VALID', name='dec4')
        
        self.caps2_4 = tf.image.resize_images(self.output2, [44, 44])
        self.dec2_4 = self.OGU1(self.caps2_4, self.dec2_5, self.Fea_P2_4, fea_dim, 44, cfg.D, 2, 4)
        
        self.caps2_3 = tf.image.resize_images(self.caps2_4, [88, 88])
        self.dec2_3 = self.OGU1(self.caps2_3, self.dec2_4, self.Fea_P2_3, fea_dim, 88, cfg.D, 2, 3)
        
        self.caps2_2 = tf.image.resize_images(self.caps2_3, [176, 176])
        self.dec2_2 = self.OGU1(self.caps2_2, self.dec2_3, self.Fea_P2_2, fea_dim, 176, cfg.D, 2, 2)
        
        self.caps2_1 = tf.image.resize_images(self.caps2_2, [352, 352])
        self.dec2_1 = self.OGU1(self.caps2_1, self.dec2_2, self.Fea_P2_1, fea_dim, 352, cfg.D, 2, 1)            
        
        self.output_Score2 = self.Conv_2d(self.dec2_1, [1, 1, fea_dim, 2], 0.01, padding='VALID', name='output_Score2')
        self.Score = self.output_Score1_1 + self.output_Score2
            
        self.Score = tf.reshape(self.Score, [-1,2])

        self.Prob = tf.clip_by_value(tf.nn.softmax(self.Score), 1e-8, 1.0)

        #Get the contour term
        self.Prob_C = tf.reshape(self.Prob, [1, 352, 352, 2])
        self.Prob_Grad = tf.tanh(self.im_gradient(self.Prob_C))
        self.Prob_Grad = tf.tanh(tf.reduce_sum(self.im_gradient(self.Prob_C), reduction_indices=3, keep_dims=True))

        self.label_C = tf.reshape(self.label_holder, [1, 352, 352, 2])
        self.label_Grad = tf.cast(tf.greater(self.im_gradient(self.label_C), self.contour_th), tf.float32)
        self.label_Grad = tf.cast(tf.greater(tf.reduce_sum(self.im_gradient(self.label_C),
                                                           reduction_indices=3, keep_dims=True),
                                             self.contour_th), tf.float32)

        self.C_IoU_LOSS = self.Loss_IoU(self.Prob_Grad, self.label_Grad)

        #self.Contour_Loss = self.Loss_Contour(self.Prob_Grad, self.label_Grad)

        #Loss Function
        self.Loss_Mean = self.C_IoU_LOSS + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score, labels=self.label_holder))
                                                                                  
        #self.Loss_Mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score, labels=self.label_holder))

        self.correct_prediction = tf.equal(tf.argmax(self.Score,1), tf.argmax(self.label_holder, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        ##Loss for Score1
        #self.Score1_1 = tf.reshape(self.output_Score1_1, [-1,2])
    
        #self.Prob1 = tf.nn.softmax(self.Score1_1)
    
        ##Get the contour term
        #self.Prob_C1 = tf.reshape(self.Prob1, [1, 352, 352, 2])
        #self.Prob_Grad1 = tf.tanh(self.im_gradient(self.Prob_C1))
        #self.Prob_Grad1 = tf.tanh(tf.reduce_sum(self.im_gradient(self.Prob_C1), reduction_indices=3, keep_dims=True))
    
        #self.C_IoU_LOSS1 = self.Loss_IoU(self.Prob_Grad1, self.label_Grad)
    
        ##self.Contour_Loss = self.Loss_Contour(self.Prob_Grad, self.label_Grad)
    
        ##Loss Function
        ##self.Loss_Mean1 = self.C_IoU_LOSS1 + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score1_1, labels=self.label_holder))
        #self.Loss_Mean1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score1_1, labels=self.label_holder))
        ##self.Loss_Mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score, labels=self.label_holder))
    
        #self.correct_prediction1 = tf.equal(tf.argmax(self.Score1_1,1), tf.argmax(self.label_holder, 1))
        #self.accuracy1 = tf.reduce_mean(tf.cast(self.correct_prediction1, tf.float32))
        
        ##Loss for Score2
        #self.Score2 = tf.reshape(self.output_Score2, [-1,2])
    
        #self.Prob2 = tf.nn.softmax(self.Score2)
    
        ##Get the contour term
        #self.Prob_C2 = tf.reshape(self.Prob2, [1, 352, 352, 2])
        #self.Prob_Grad2 = tf.tanh(self.im_gradient(self.Prob_C2))
        #self.Prob_Grad2 = tf.tanh(tf.reduce_sum(self.im_gradient(self.Prob_C2), reduction_indices=3, keep_dims=True))
    
    
        #self.C_IoU_LOSS2 = self.Loss_IoU(self.Prob_Grad2, self.label_Grad)
    
        ##self.Contour_Loss = self.Loss_Contour(self.Prob_Grad, self.label_Grad)
    
        ##Loss Function
        ##self.Loss_Mean2 = self.C_IoU_LOSS2 + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score2, labels=self.label_holder))
        #self.Loss_Mean2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score2, labels=self.label_holder))
        ##self.Loss_Mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score, labels=self.label_holder))
    
        #self.correct_prediction2 = tf.equal(tf.argmax(self.Score2,1), tf.argmax(self.label_holder, 1))
        #self.accuracy2 = tf.reduce_mean(tf.cast(self.correct_prediction2, tf.float32))        

    def Conv_2d(self, input_, shape, stddev, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)

            b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            conv = tf.nn.bias_add(conv, b)

            return conv
    
    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            #tf.add_to_collection('conv_out', relu)
            return relu    
        
    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)    

    def Deconv_2d(self, input_, output_shape,
                  k_s=3, st_s=2, stddev=0.01, padding='SAME', name="deconv2d"):
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                shape=[k_s, k_s, output_shape[3], input_.get_shape()[3]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            deconv = tf.nn.conv2d_transpose(input_, W, output_shape=output_shape,
                                            strides=[1, st_s, st_s, 1], padding=padding)

            b = tf.get_variable('b', [output_shape[3]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, b)

        return deconv

    def Contrast_Layer(self, input_, k_s=3):
        h_s = int(k_s / 2)
        return tf.subtract(input_, tf.nn.avg_pool(tf.pad(input_, [[0, 0], [h_s, h_s], [h_s, h_s], [0, 0]], 'SYMMETRIC'),
                                                  ksize=[1, k_s, k_s, 1], strides=[1, 1, 1, 1], padding='VALID'))

    def sobel_filter(self):
        fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
        fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)

        fx = np.stack((fx, fx), axis=2)
        fy = np.stack((fy, fy), axis=2)

        fx = np.reshape(fx, (3, 3, 2, 1))
        fy = np.reshape(fy, (3, 3, 2, 1))

        tf_fx = tf.Variable(tf.constant(fx))
        tf_fy = tf.Variable(tf.constant(fy))

        return tf_fx, tf_fy

    def im_gradient(self, im):
        gx = tf.nn.depthwise_conv2d(tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC'),
                                    self.sobel_fx, [1, 1, 1, 1], padding='VALID')
        gy = tf.nn.depthwise_conv2d(tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC'),
                                    self.sobel_fy, [1, 1, 1, 1], padding='VALID')
        return tf.sqrt(tf.add(tf.square(gx), tf.square(gy)))

    def Loss_IoU(self, pred, gt):
        inter = tf.reduce_sum(tf.multiply(pred, gt))
        union = tf.add(tf.reduce_sum(tf.square(pred)), tf.reduce_sum(tf.square(gt)))

        if inter == 0:
            return 0
        else:
            return 1 - (2*(inter+1)/(union + 1))

    def Loss_Contour(self, pred, gt):
        return tf.reduce_mean(-gt*tf.log(pred+0.00001) - (1-gt)*tf.log(1-pred+0.00001))

    def L2(self, tensor, wd=0.0005):
        return tf.mul(tf.nn.l2_loss(tensor), wd, name='L2-Loss')
    
    def dilation(self, input, input_dim, output_dim, name):
        with tf.variable_scope(name) as scope:
            a = tf.nn.relu(self.Atrous_conv2d(input, [3, 3, input_dim, output_dim], 1, 0.01, name = 'dilation1'))
            b = tf.nn.relu(self.Atrous_conv2d(input, [3, 3, input_dim, output_dim], 3, 0.01, name = 'dilation3'))
            c = tf.nn.relu(self.Atrous_conv2d(input, [3, 3, input_dim, output_dim], 5, 0.01, name = 'dilation5'))
            d = tf.nn.relu(self.Atrous_conv2d(input, [3, 3, input_dim, output_dim], 7, 0.01, name = 'dilation7'))
            e = tf.concat([a, b, c, d], axis = 3)
            
        return e   
    
    
    def Atrous_conv2d(self, input, shape, rate, stddev, name, padding = 'SAME'):
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                shape = shape,
                                initializer = tf.truncated_normal_initializer(stddev = stddev))
            atrous_conv = tf.nn.atrous_conv2d(input, W, rate = rate, padding = padding)
            b = tf.get_variable('b', shape = [shape[3]], initializer = tf.constant_initializer(0.0))
            atrous_conv = tf.nn.bias_add(atrous_conv, b)
            
        return atrous_conv    
    
    def OGU1(self, input1, input2, input3, channel1, data_size, channel2, i, j):
        self.caps3, self.dec4, self.Fea_P3, fea_dim, 88, cfg.D, 1, 3
        output_Up = tf.nn.relu(self.Deconv_2d(input2, [1, data_size, data_size, channel1], 5, 2, name='output_Deconv' + str(i) + str(j)))
        con = self.Conv_2d(tf.concat([output_Up, input3], axis=3), [1, 1, channel1 + channel1, channel1], 0.01, padding='VALID', name='Fea_Concat'+ str(i) + str(j))
        c = self.Conv_2d(con, [1, 1, channel1, channel1], 0.01, padding='VALID', name='Fea_Concat1'+ str(i) + str(j))
        c_channel = self.Conv_2d(c, [1, 1, channel1, 1], 0.01, padding='VALID', name='Fea_Concat1_channel'+ str(i) + str(j))
        c1 = tf.reshape(c, shape=[data_size * data_size, channel1])
        c1_channel = tf.reshape(c_channel, shape=[data_size * data_size, 1])
        c2 = tf.transpose(c1, perm=[1, 0])
        a = tf.reshape(tf.nn.sigmoid(tf.matmul(c2, c1_channel)), shape=[1, 1, 1, channel1])
        a_tile = tf.tile(a, [1, data_size, data_size, 1])
        #conr = tf.reshape(con, shape=[data_size * data_size, 1])
        ca = tf.multiply(con, a_tile)
        cona = self.Conv_2d(tf.reshape(ca, shape=[1, data_size, data_size, channel1]), [1, 1, channel1, channel1], 0.01, padding='VALID', name='cona4'+ str(i) + str(j))
        #capsa = tf.nn.sigmoid(self.Conv_2d(input1, [1, 1, channel2, 1], 0.01, padding='VALID', name='capsa4_1'+ str(i) + str(j)))
        dec = self.Conv_2d(tf.concat([cona, input1], axis=3), [1, 1, channel1 + channel2, channel1], 0.01, padding='VALID', name='Fea_Concat2'+ str(i) + str(j))
        #dec = self.Conv_2d(tf.multiply(cona, capsa), [1, 1, channel1, channel1], 0.01, padding='VALID', name='Dec4_1'+ str(i) + str(j))        
        return dec
    
    def OGU2(self, input1, input2, input3, channel1, data_size, channel2, i, j):
        output_Up = tf.nn.relu(self.Deconv_2d(input2, [1, v, data_size, channel1], 5, 2, name='output_Deconv' + str(i) + str(j)))
        con = self.Conv_2d(tf.concat([output_Up, input3], axis=3), [1, 1, channel1 + channel1, channel1], 0.01, padding='VALID', name='Fea_Concat'+ str(i) + str(j))
        #c = self.Conv_2d(con, [1, 1, channel1, channel1], 0.01, padding='VALID', name='Fea_Concat1'+ str(i) + str(j))
        #c1 = tf.reshape(c, shape=[data_size * data_size, channel1])
        #c2 = tf.transpose(c1, perm=[1, 0])
        #a = tf.nn.sigmoid(tf.matmul(c1, c2))
        #conr = tf.reshape(con, shape=[data_size * data_size, channel1])
        #ca = tf.matmul(a, conr)
        #cona = self.Conv_2d(tf.reshape(ca, shape=[1, data_size, data_size, channel1]), [1, 1, channel1, channel1], 0.01, padding='VALID', name='cona4'+ str(i) + str(j))
        capsa = tf.nn.sigmoid(self.Conv_2d(input1, [1, 1, channel2, 1], 0.01, padding='VALID', name='capsa4_1'+ str(i) + str(j)))
        dec = self.Conv_2d(tf.multiply(con, capsa), [1, 1, channel1, channel1], 0.01, padding='VALID', name='Dec4_1'+ str(i) + str(j))        
        return dec    
    
    def SWG1(self, input1, input2, channel1, channel2, channel3, i, j):
        dec_F = self.Conv_2d(tf.concat([input1, input2], axis=3), [1, 1, channel1 + channel2, channel3], 0.01, padding='VALID', name='dec_F'+ str(i) + str(j))
        output_Score = self.Conv_2d(input1, [1, 1, channel1, 2], 0.01, padding='VALID', name='output_Score'+ str(i) + str(j))
        output_Score_S = tf.split(output_Score, num_or_size_splits=2,axis=3)
        enc1 = tf.multiply(tf.tile(output_Score_S[0], (1, 1, 1, channel3)), dec_F)
        enc2 = self.Conv_2d(tf.concat([output_Score_S[0], dec_F], axis=3), [1, 1, 1 + channel3, channel3], 0.01, padding='VALID', name='Attention'+ str(i) + str(j))
        enc = enc1 + enc2 + dec_F
        pool = self._max_pool(enc, 'pool'+ str(i) + str(j))    
        return pool, enc, output_Score, output_Score_S
    
    def SWG2(self, input1, input2, input3, data_size, channel1, channel2, channel3, i, j):
        dec_F = self.Conv_2d(tf.concat([input1, input2], axis=3), [1, 1, channel1 + channel2, channel3], 0.01, padding='VALID', name='dec_F'+ str(i) + str(j))        
        #self.output_Score1_2 = self.Conv_2d(self.dec2, [1, 1, fea_dim, 2], 0.01, padding='VALID', name='output_Score1_2')
        output_Score_S = tf.image.resize_images(input3, [data_size, data_size])        
        enc1 = tf.multiply(tf.tile(output_Score_S, (1, 1, 1, channel3)), dec_F)
        enc2 = self.Conv_2d(tf.concat([output_Score_S, dec_F], axis=3), [1, 1, 1 + channel3, channel3], 0.01, padding='VALID', name='Attention'+ str(i) + str(j))
        enc = enc1 + enc2 + dec_F
        pool = self._max_pool(enc, 'pool'+ str(i) + str(j)) 
        return pool, enc, output_Score_S    
    
    
    def kernel_tile(self, input, kernel, stride):
        # output = tf.extract_image_patches(input, ksizes=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='VALID')
    
        input_shape = input.get_shape()
        tile_filter = np.zeros(shape=[kernel, kernel, input_shape[3],
                                      kernel * kernel], dtype=np.float32)
        for i in range(kernel):
            for j in range(kernel):
                tile_filter[i, j, :, i * kernel + j] = 1.0
    
        tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)
        output = tf.nn.depthwise_conv2d(input, tile_filter_op, strides=[
                                        1, stride, stride, 1], padding='SAME')
        output_shape = output.get_shape()
        output = tf.reshape(output, shape=[int(output_shape[0]), int(
            output_shape[1]), int(output_shape[2]), int(input_shape[3]), kernel * kernel])
        output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
    
        return output     
    
    
    def mat_transform(self, input, caps_num_c, regularizer, tag=False):
        batch_size = int(input.get_shape()[0])
        caps_num_i = int(input.get_shape()[1])
        output = tf.reshape(input, shape=[batch_size, caps_num_i, 1, 4, 4])
        # the output of capsule is miu, the mean of a Gaussian, and activation, the sum of probabilities
        # it has no relationship with the absolute values of w and votes
        # using weights with bigger stddev helps numerical stability
        w = slim.variable('w', shape=[1, caps_num_i, caps_num_c, 4, 4], dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0),
                          regularizer=regularizer)
    
        w = tf.tile(w, [batch_size, 1, 1, 1, 1])
        output = tf.tile(output, [1, 1, caps_num_c, 1, 1])
        votes = tf.reshape(tf.matmul(output, w), [batch_size, caps_num_i, caps_num_c, 16])
    
        return votes
    
    def em_routing(self, votes, activation, caps_num_c, regularizer, tag=False):
        test = []
    
        batch_size = int(votes.get_shape()[0])
        caps_num_i = int(activation.get_shape()[1])
        n_channels = int(votes.get_shape()[-1])
    
        sigma_square = []
        miu = []
        activation_out = []
        beta_v = slim.variable('beta_v', shape=[caps_num_c, n_channels], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0),#tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                               regularizer=regularizer)
        beta_a = slim.variable('beta_a', shape=[caps_num_c], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0),#tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                               regularizer=regularizer)
    
        # votes_in = tf.stop_gradient(votes, name='stop_gradient_votes')
        # activation_in = tf.stop_gradient(activation, name='stop_gradient_activation')
        votes_in = votes
        activation_in = activation
    
        for iters in range(cfg.iter_routing):
            # if iters == cfg.iter_routing-1:
    
            # e-step
            if iters == 0:
                r = tf.constant(np.ones([batch_size, caps_num_i, caps_num_c], dtype=np.float32) / caps_num_c)
            else:
                # Contributor: Yunzhi Shi
                # log and exp here provide higher numerical stability especially for bigger number of iterations
                log_p_c_h = -tf.log(tf.sqrt(sigma_square)) - \
                            (tf.square(votes_in - miu) / (2 * sigma_square))
                log_p_c_h = log_p_c_h - \
                            (tf.reduce_max(log_p_c_h, axis=[2, 3], keep_dims=True) - tf.log(10.0))
                p_c = tf.exp(tf.reduce_sum(log_p_c_h, axis=3))
    
                ap = p_c * tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])
    
                # ap = tf.reshape(activation_out, shape=[batch_size, 1, caps_num_c])
    
                r = ap / (tf.reduce_sum(ap, axis=2, keep_dims=True) + cfg.epsilon)
    
            # m-step
            r = r * activation_in
            r = r / (tf.reduce_sum(r, axis=2, keep_dims=True)+cfg.epsilon)
    
            r_sum = tf.reduce_sum(r, axis=1, keep_dims=True)
            r1 = tf.reshape(r / (r_sum + cfg.epsilon),
                            shape=[batch_size, caps_num_i, caps_num_c, 1])
    
            miu = tf.reduce_sum(votes_in * r1, axis=1, keep_dims=True)
            sigma_square = tf.reduce_sum(tf.square(votes_in - miu) * r1,
                                         axis=1, keep_dims=True) + cfg.epsilon
    
            if iters == cfg.iter_routing-1:
                r_sum = tf.reshape(r_sum, [batch_size, caps_num_c, 1])
                cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square,
                                                             shape=[batch_size, caps_num_c, n_channels])))) * r_sum
    
                activation_out = tf.nn.softmax(cfg.ac_lambda0 * (beta_a - tf.reduce_sum(cost_h, axis=2)))
            else:
                activation_out = tf.nn.softmax(r_sum)
            # if iters <= cfg.iter_routing-1:
            #     activation_out = tf.stop_gradient(activation_out, name='stop_gradient_activation')
    
        return miu, activation_out, test
    


if __name__ == "__main__":

    img = cv2.imread("E:/LY/NLDF-master/dataset/img/1.jpg")

    h, w = img.shape[0:2]
    img = cv2.resize(img, (img_size,img_size)) - vgg16.VGG_MEAN
    img = img.reshape((1, img_size, img_size, 3))

    label = cv2.imread("E:/LY/NLDF-master/dataset/label/1.png")[:, :, 0]
    label = cv2.resize(label, (label_size, label_size))
    label = label.astype(np.float32) / 255
    label = np.stack((label, 1-label), axis=2)
    label = np.reshape(label, [-1, 2])

    sess = tf.Session()

    model = Model()
    model.build_model()

    max_grad_norm = 1
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.C_IoU_LOSS, tvars), max_grad_norm)
    opt = tf.train.AdamOptimizer(1e-5)
    optimizer = opt.apply_gradients(zip(grads, tvars))

    sess.run(tf.global_variables_initializer())

    for i in range(200):  #python2.x xrange, python3.x range
        _, C_IoU_LOSS = sess.run([optimizer, model.C_IoU_LOSS],
                                 feed_dict={model.input_holder: img,
                                            model.label_holder: label})

        print('[Iter %d] Contour Loss: %f' % (i, C_IoU_LOSS))

    boundary, gt_boundary = sess.run([model.Prob_Grad, model.label_Grad],
                                     feed_dict={model.input_holder: img,
                                                model.label_holder: label})

    boundary = np.squeeze(boundary)
    boundary = cv2.resize(boundary, (w, h))

    gt_boundary = np.squeeze(gt_boundary)
    gt_boundary = cv2.resize(gt_boundary, (w, h))

    cv2.imshow('boundary', np.uint8(boundary*255))
    cv2.imshow('boundary_gt', np.uint8(gt_boundary*255))

    cv2.waitKey()
