import rospy
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
from collections import deque
import csv
import os
from math import pi
import pandas as pd
import matplotlib.image as mpimg
def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'test_dataset.csv'), names=['center', 'speed', 'steering'])

    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    X = data_df['center'].values
    #and our steering commands as our output data
    y = data_df['steering'].values

    return X, y

def make_predictor(cnn_json_path, cnn_weights_path,lstm_json_path,lstm_weights_path):
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
    return(steering_angle)



if __name__ == '__main__':
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Save predicted angles to csv')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='test_dataset')
    args = parser.parse_args()

    cnn_json_path = 'cnn.json'
    cnn_weights_path = 'cnn.weights'
    lstm_json_path = 'lstm.json'
    lstm_weights_path = 'lstm.weights'
    
    model = make_predictor(cnn_json_path,cnn_weights_path,lstm_json_path,lstm_weights_path)
    # node = SteeringNode()

    # rospy.Timer(rospy.Duration(1), process(model, node.img))

    # rospy.spin()
    X, true_y = load_data(args)
    predicted_list = list()
    for image in X:
        img = mpimg.imread(os.path.join(args.data_dir, image))
        pre_angle = process(model, img)
        predicted_list.append(pre_angle)


        text_file = open("test_dataset/predict.csv", "w")
    for row in predicted_list:
        
        text_file.write(str(row) +'\n')
    text_file.close()
    print("predicted angles saved!")
    