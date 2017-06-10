#!/usr/bin/env python

from __future__ import print_function

# Math/Data structures/Functions
import numpy as np
from math import sin, cos, tanh
from Transformations import quat_to_euler
import Image
from Map import Map

# ROS
import rospy
import message_filters
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

### Global Variables
map_ = None
res_ = None
origin_ = None


def save_map_to_img(range=15, file_name='grid_map'):
    '''
    Creates a new image and saves the given map data in it.
    :param size: The size of the image in Pixels.
    :param data: The map-data as a numpy-array.
    :param range: The data range of the map values.
    :param file_name: The file-name for the saved image.
    :return: None
    '''
    global map_
    # Invert map_
    map_ = np.zeros(map_.shape)
    # Create new image
    img = Image.new('L', map_.shape)
    # Save map data to image
    img.putdata(map_.flatten(), 1, 0)
    # Save image to disk
    img.save('/home/robin/catkin_ws/src/int_agents_project/data/' + file_name + '.png')
    img.show()
    
    print('Map saved successfully, program shutting down.')


def cb_map(msg):
    global map_, res_, origin_
    
    if map_ is not None:
        return
    map_ = Map((msg.info.width, msg.info.height), values=msg.data, resolution=msg.info.resolution, origin=[msg.info.origin.position.x, msg.info.origin.position.y])
    
    # map_.save_map_to_img()
    

def get_numpy_pose(ros_pose):
    return np.array([ros_pose.position.x, ros_pose.position.y, quat_to_euler(ros_pose.orientation)])


def cb_scan_pose_goal(scan, odom):
    print('Got a scan/pose!')
    
    # Build RGB Image from map, sensor and pose data
    pose = get_numpy_pose(odom.pose.pose)
    
    if map_ is not None:
        map_.set_robot_position(pose)
        map_.set_scan(scan)

        map_.draw_all()
        map_.save_map_to_img()
        
        # Publish occupancy grid for map
        occ_grid = OccupancyGrid()
        origin = Pose()
        origin.orientation.w = 1.0
        occ_grid.header.frame_id = 'map'
        occ_grid.info.height = map_.size[0]
        occ_grid.info.width = map_.size[1]
        occ_grid.info.origin = origin
        occ_grid.info.resolution = map_.resolution
        occ_grid.data = (map_.values/254*100).flatten()
        
        pub_map.publish(occ_grid)
    
    return


def cb_goal(msg):
    print('Got a goal!')
    
    goal = get_numpy_pose(msg.pose)
    
    if map_ is not None:
        # map_.set_robot_position(pose)
        map_.set_goal(goal[:2])

        map_.save_map_to_img()
    
    return


if __name__ == '__main__':
    ##### ROS #####
    # Initialize this node
    rospy.init_node('gather_data', anonymous=True)
    
    # Subscriber
    sub_sensor_data = rospy.Subscriber('/map', OccupancyGrid, cb_map)
    sub_pose_data = rospy.Subscriber('/move_base_simple/goal', PoseStamped, cb_goal)

    sub_scan = message_filters.Subscriber('/scan', LaserScan)
    sub_pose = message_filters.Subscriber('/amcl_pose', PoseWithCovarianceStamped)
    sub_goal = message_filters.Subscriber('/move_base_simple/goal', PoseStamped)
    
    ts = message_filters.TimeSynchronizer([sub_scan, sub_pose], 10)
    ts.registerCallback(cb_scan_pose_goal)
    
    # Publisher
    pub_map = rospy.Publisher('/ia/map', OccupancyGrid, queue_size=1)
    
    # ROS spin
    # rospy.spin()
    rate = rospy.Rate(10)  # 10hz
    
    while(True):
        rate.sleep()
    