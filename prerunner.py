import rospy
from SteeringNode import SteeringNode
import argparse
import json
from scipy import misc
from keras.optimizers import SGD
from keras.models import model_from_json, load_model
import utils
import numpy as np
import thread
import tensorflow as tf
from geometry_msgs.msg import Twist
import time
from premodel import ChauffeurModel

from keras import backend as K
from keras import metrics
from keras.models import load_model
from keras.models import model_from_json
import argparse
from collections import deque

from scipy import misc
import csv
import os
from math import pi
import pandas as pd
def make_predictor():
        K.set_learning_phase(0)
        model = ChauffeurModel(
            cnn_json_path,
            cnn_weights_path,
            lstm_json_path,
            lstm_weights_path)
        return model.make_stateful_predictor()


def process(predictor, img):
    steering_angle = predictor(img)
    print(steering_angle)
    pub_steering(steering_angle)


def pub_steering(steering):
    move_cmd = Twist()
    move_cmd.linear.x = 0.5
    move_cmd.angular.z = steering
    # move_cmd.angular.z = 0
    node.pub.publish(move_cmd)

if __name__ == '__main__':

    cnn_json_path = 'cnn.json'
    cnn_weights_path = 'cnn.weights'
    lstm_json_path = 'lstm.json'
    lstm_weights_path = 'lstm.weights'
    
    model = make_predictor()
    node = SteeringNode()

    # rospy.Timer(rospy.Duration(1), process(model, node.img))

    # rospy.spin()

    while not rospy.is_shutdown():     
            time.sleep(0.01)
            if node.img is not None:
                # r
                # save both data every one second.
                process(model, node.img)
            else:  
                rospy.loginfo("no image received")