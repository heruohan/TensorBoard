# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 23:45:21 2018

@author: hecongcong
"""

import os.path
import re
import time
import numpy as np
import tensorflow as tf
from tensorflow.models.tutorials.image.cifar10 import cifar10


batch_size=128
max_steps=1000000
num_gpus=4


####定义计算损失的函数tower_loss.
def tower_loss(scope):
    images,labels=cifar10.distorted_inputs()
    logits=cifar10.inference(images)
    _=cifar10.loss(logits,labels)
    losses=tf.get_collection('losses',scope)
    total_loss=tf.add_n(losses,name='total_loss')
    return(total_loss)


####定义函数average_gradients,负责将不同GPU计算出的梯度进行合成.
def average_gradients(tower_grads):
    average_grads=[]
    for grad_and_vars in zip(*tower_grads):
        grads=[]
        for g,_ in grad_and_vars:
            expanded_g=tf.expand_dims(g,0) #g=10,则ouput为[10].
            grads.append(expanded_g)
        
        grad=tf.concat(grads,0) #grad=[[10],[10]],ouput为[10,10]
        grad=tf.reduce_mean(grad,0)
        v=grad_and_vars[0][1]
        grad_and_var=(grad,v)
        average_grads.append(grad_and_var)
    return(average_grads)


####定义训练的函数.
def train():
    with tf.Graph().as_default(),tf.device('/cpu:0'):
        global_step=tf.get_variable('global_step',[],\
                                initializer=tf.constant_initializer(0),\
                                trainable=False) #output为0.0
        
        num_batches_per_epoch=cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN//batch_size
        '''
        解释(例子)：
        for i in range(epoch):
            for j in range(batch):
                shape(feed_dict)=batch_size
        则：
        1.一个epoch对应batch个数.
        2.一个epoch对应batch*batch_size个样本(Examples).
        3.一个epoch对应的batch数=一个epoch对应的example数/batch_size.
        '''
        decay_steps=int(num_batches_per_epoch*cifar10.NUM_EPOCHS_PER_DECAY)
        '''
        decay_steps=batch*epochs
        '''
        lr=tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,\
                                      global_step,\
                                      decay_steps,\
                                      cifar10.LEARNING_RATE_DECAY_FACTOR,\
                                      staircase=True)
        '''
        创建随训练步数衰减的学习速率.
        '''
        opt=tf.train.GradientDescentOptimizer(lr)
        
        ####定义储存各GPU计算结果的列表tower_grads.
        tower_grads=[]
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME,i)) as scope:
                    
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            








