# __author__ = "Shiyu Mou"
# __copyright__ = "Copyright 2017, Iquotient Robotics"
# __version__ = "0.0.2"
# __email__ = "fdevmsy@gmail.com"

# Usage: pthon record_data.py. This script will record your webcam data (saved as paths to images) 
# and the Twist data ([Speed, Steering angle]) to a csv file
# Check your camera topic and velocity topic before running this script. 

from __future__ import print_function
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import roslib
import tf.transformations
from geometry_msgs.msg import Twist
import time
import os

class Record_data:
    def __init__(self):
        self.imgList= list()
        self.steeringList = list()
        self.count = 0
        self.steering_data = [0, 0]

    def launch(self):

        self.bridge = CvBridge() 
        img_topic = "/usb_cam/image_raw"
        rospy.Subscriber(img_topic, Image, self.callback1)
        velocity_topic = "cmd_vel_mux/input/teleop"
        rospy.Subscriber(velocity_topic, Twist, self.callback2)

    def callback1(self, data):

        # Convert image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        global image_data
        image_data = cv_image


    def callback2(self, msg):
        self.steering_data = [msg.linear.x, msg.angular.z]

        
    def take_picture(self):
       
        # Save an image
        img_title = 'photo' + str(self.count) + '.jpg'
        cv2.imwrite("dataset/images/"+img_title, image_data)
        self.imgList.append(img_title)
        
        self.count = self.count + 1
        rospy.loginfo(self.count)


    def record_steering(self):
            rospy.loginfo(self.steering_data)
            self.steeringList.append(self.steering_data)

if __name__ == '__main__':

    # The dataset will be saved to "dataset/images". 
    if not os.path.exists("dataset/images"):
        os.makedirs("dataset/images")
    # Initialize
    rospy.init_node('record_data', anonymous=False)

    data = Record_data()  
    data.launch()

    # break when ctrl + c is pressed
    while not rospy.is_shutdown():     
            time.sleep(0.05)
            # save both data every one second.
            data.take_picture()
            data.record_steering()

    # save to csv file 
    dataset = list()
    rospy.loginfo("Saving dataset...")
    for img_path, (speed, angle) in zip(data.imgList, data.steeringList):
        newString = "images/" + str(img_path) + ", " + str(speed) + ", " + str(angle)
        dataset.append(newString)
    # rospy.loginfo(dataset)
    # rospy.loginfo(imgList)
    # rospy.loginfo(len(imgList))
    # rospy.loginfo(steeringList)
    # rospy.loginfo(len(steeringList))
    # print result to text file
    text_file = open("dataset/dataset.csv", "w")
    for row in dataset:
        
        text_file.write(str(row) +'\n')
    text_file.close()
    rospy.loginfo("Dataset saved!")
    rospy.sleep(0.5)
