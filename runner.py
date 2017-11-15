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
def process(model, img):
    if img is not None:
        # print(img)
        # print(img.shape)
        img = misc.imresize(img[:, :, :], (66, 200, 3))
        # print(img.shape)
        img = utils.rgb2yuv(img)
        # print(img.shape)
        img = np.array([img])
        # print(img.shape)
        # print('\n\n')
        # steering_angle = model.predict(img[None, :, :, :])[0][0]
        steering_angle = float(model.predict(img, batch_size=1))
        print(steering_angle)
        pub_steering(steering_angle)


def get_model(model_file):

    with open(model_file, 'r') as jfile:

        model = model_from_json(jfile.read())

    # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(sgd, "mse")
    model.compile("adam", "mse")

    weights_file = model_file.replace('json', 'h5')
    model.load_weights(weights_file)
    
    # return model
    # graph = tf.get_default_graph()
    return model

def pub_steering(steering):
    move_cmd = Twist()
    move_cmd.linear.x = 0.8
    move_cmd.angular.z = steering

    node.pub.publish(move_cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Runner')
    parser.add_argument('model', type=str, help='Path to model definition json. \
                        Model weights should be on the same path.')
    
    args = parser.parse_args()
    model = get_model(args.model)
    node = SteeringNode()

    # rospy.Timer(rospy.Duration(1), process(model, node.img))

    # rospy.spin()

    while not rospy.is_shutdown():     
            time.sleep(0.01)
            # save both data every one second.
            process(model, node.img)