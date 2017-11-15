import threading
import numpy as np
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class SteeringNode(object):
    def __init__(self):
        rospy.init_node('steering_model')
        # self.model = get_model_callback()
        # self.model = get_model_callback
        # self.get_model = get_model_callback
        # self.predict = model_callback
        self.img =  None
        self.steering = 0.
        # self.image_lock = threading.RLock()
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image,
                                          self.update_image)
        # self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image,
        #                                   self.update_image)
        # self.pub = rospy.Publisher('/mobile_base/mobile_base_controller/cmd_vel',
        #                            Twist, queue_size=1)
        self.pub = rospy.Publisher('cmd_vel_mux/input/teleop',
                           Twist, queue_size=1)
        # rospy.Timer(rospy.Duration(1), self.get_steering)
        # rospy.spin()
    def update_image(self, img):
        d = map(ord, img.data)
        arr = np.ndarray(shape=(img.height, img.width, 3),
                         dtype=np.int,
                         buffer=np.array(d))[:,:,::-1]
        self.img = arr
        # if self.image_lock.acquire(True):
        #     self.img = arr
            # if self.model is None:
            #     self.model = self.get_model()
            # self.steering = self.predict(self.model, self.img)
            # self.image_lock.release()

    def get_steering(self, event):
        if self.img is None:
            return
        move_cmd = Twist()
        move_cmd.linear.x = 0.3
        move_cmd.angular.z = self.steering

        self.pub.publish(move_cmd)