#!/usr/bin/env python

from __future__ import print_function

# Math/Data structures/Functions
import numpy as np
from scipy.spatial.distance import euclidean
from math import sin, cos, tanh
from Transformations import quat_to_euler, tf_world_to_robot, angle_between_vectors, get_numpy_pose
import Image
from Map import Map
from threading import Lock
from time import sleep

# ROS
import rospy
import message_filters
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from std_srvs.srv import Empty

from time import sleep

### Global Variables


class SimulationEnvironment:
    # ROS - Publisher
    pub_map = None
    # ROS - Service
    srv_step = None
    
    map = None
    res = None
    origin = None
    
    pose = None
    goal = None
    scan = None
    
    last_action = None
    new_state = False
    
    state_mutex = Lock()
    
    def __init__(self, goal):
        ##### ROS #####
        # Initialize this node
        rospy.init_node('gather_data', anonymous=True)
    
        # Subscriber
        sub_sensor_data = rospy.Subscriber('/map', OccupancyGrid, self.cb_map)
        # sub_pose_data = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.cb_goal)
        # sub_cmd_vel = rospy.Subscriber('/cmd_vel_mux/input/teleop', Twist, self.cb_cmd_vel)

        sub_scan = message_filters.Subscriber('/scan', LaserScan)
        sub_pose = message_filters.Subscriber('/base_pose_ground_truth', Odometry)
    
        ts = message_filters.TimeSynchronizer([sub_scan, sub_pose], 10)
        ts.registerCallback(self.cb_scan_pose)
    
        # Publisher
        self.pub_map = rospy.Publisher('/ia/map', OccupancyGrid, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)

        self.goal = goal

    def cb_cmd_vel(self, msg):
        self.last_action = msg
        
    def cb_map(self, msg):
        if self.map is not None:
            return
        self.map = Map((msg.info.width, msg.info.height), values=msg.data, resolution=msg.info.resolution,
                   origin=[msg.info.origin.position.x, msg.info.origin.position.y])
        self.map.set_goal(self.goal)
        
        # self.map.save_map_to_img()
    
    def cb_scan_pose(self, scan, odom):
        # print('Got a scan/pose!')
        
        # Build RGB Image from map, sensor and pose data
        self.pose = get_numpy_pose(odom.pose.pose)
        self.scan = scan
        
        if self.map is not None:
            self.map.set_robot_position(self.pose)
            self.map.set_scan(self.scan)
            
            if self.map.draw_all():
                self.map.save_map_to_img()
                
                # Publish occupancy grid for map
                occ_grid = OccupancyGrid()
                origin = Pose()
                origin.orientation.w = 1.0
                occ_grid.header.frame_id = 'map'
                occ_grid.info.height = self.map.size[0]
                occ_grid.info.width = self.map.size[1]
                occ_grid.info.origin = origin
                occ_grid.info.resolution = self.map.resolution
                occ_grid.data = (self.map.values / 254 * 100).flatten()
                
                self.pub_map.publish(occ_grid)
        
        self.state_mutex.acquire()
        self.new_state = True
        self.state_mutex.release()
        
        return
    
    def cb_goal(self, msg):
        print('Got a goal!')
        
        self.goal = get_numpy_pose(msg.pose)
        
        if self.map is not None:
            # self.map.set_robot_position(pose)
            self.map.set_goal(self.goal[:2])
            
            self.map.save_map_to_img()
        
        return
    
    def save_map_to_img(self, range=15, file_name='grid_map'):
        '''
        Creates a new image and saves the given map data in it.
        :param size: The size of the image in Pixels.
        :param data: The map-data as a numpy-array.
        :param range: The data range of the map values.
        :param file_name: The file-name for the saved image.
        :return: None
        '''
        # Invert self.map
        self.map = np.zeros(self.map.shape)
        # Create new image
        img = Image.new('L', self.map.shape)
        # Save map data to image
        img.putdata(self.map.flatten(), 1, 0)
        # Save image to disk
        img.save('/home/robin/catkin_ws/src/int_agents_project/data/' + file_name + '.png')
        img.show()
    
        print('Map saved successfully, program shutting down.')
        
    def step(self, actions, iterations=1):
        for i in range(iterations):
            rospy.wait_for_service('/stage/step')
            try:
                # Publish actions as cmd_vel to robot
                cmd_vel = Twist()
                cmd_vel.linear.x = actions[0]
                cmd_vel.angular.z = actions[1]
                self.last_action = cmd_vel
                self.pub_cmd_vel.publish(cmd_vel)
                # Go one step further in simulation
                step = rospy.ServiceProxy('/stage/step', Empty)
                success = step()
                return success
            except rospy.ServiceException, e:
                print('Service call failed: %s' % e)
                
    def reset(self):
        rospy.wait_for_service('/stage/reset_positions')
        try:
            step = rospy.ServiceProxy('/stage/reset_positions', Empty)
            success = step()
            return success
        except rospy.ServiceException, e:
            print('Service call failed: %s' % e)
                
    def close_to_goal(self, threshold=0.2):
        if euclidean(self.pose[:2], self.goal[:2]) < threshold:
            return True
        else:
            return False
                
    def get_state_reward(self):
        i = 0
        r = rospy.Rate(10)  # 20hz
        self.state_mutex.acquire()

        while(not self.new_state):
            self.state_mutex.release()
            sleep(0.1)
            # r.sleep()
            self.state_mutex.acquire()
            i += 1
            if i >= 50:
                self.state_mutex.release()
                return None

        self.new_state = False
        self.state_mutex.release()
        
        return self.map.values.reshape((self.map.values.shape[0], self.map.values.shape[1], 1)), self.get_reward(), self.close_to_goal()
        
    def get_reward(self, d=10, a=4, v_back=4, range_threshold=0.5):
        if self.scan is None or self.pose is None or self.goal is None:
            print('No messages received yet')
            return None
        
        # Initialize reward to 0
        reward = 0
        
        # Reward from distance to goal
        print('pose: ', self.pose)
        print('goal: ', self.goal)
        reward += d*1/euclidean(self.pose[:2], self.goal[:2])
        
        # Reward for orientation compared to goal orientation
        # Transform goal to robot frame
        goal_in_robot_frame = tf_world_to_robot(self.pose, self.goal)
        # Calculate angle between orientation of robot in its own frame and the goal in robot frame
        angular_error = angle_between_vectors(np.array([1, 0]), goal_in_robot_frame[:2])
        if angular_error > 0.5*np.pi:
            angular_error *= a
        # Add product of that with velocity direction
        reward += 2 * np.pi - angular_error
        
        # Negative reward for driving backwards
        if self.last_action is not None and self.last_action.linear.x < 0:
            reward += self.last_action.linear.x * v_back
        # reward += 0.5 * v_back

        # Reward for scan distances
        ranges = np.array(self.scan.ranges)
        short_ranges = range_threshold-ranges[ranges<range_threshold]
        reward += -np.sum(short_ranges)
        
        return reward