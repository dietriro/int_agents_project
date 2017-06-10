#!/usr/bin/env python

from numpy import (array, dot, arccos, clip, pi)
from numpy.linalg import norm
from Transformations import quat_to_euler, tf_world_to_robot
from tf.transformations import euler_from_quaternion
from threading import Thread, Lock
from SimulationEnvironment import SimulationEnvironment
import rospy
from time import sleep

# v = array([1, 0])
# u = array([-1.0, -0.1])
# c = dot(u, v)/norm(u)/norm(v) # -> cosine of the angle
# angle = arccos(clip(c, -1, 1)) # if you really want the angle
#
# # print(c)
# print (angle)
# #
# # print(euler_from_quaternion([0.0, 0.0, 0.0, 1.0]))
# #
# # r = array([2, 1, -1.57])
# # p = array([3, 2])
# #
# # print(tf_world_to_robot(r, p))
#
#
# m = Lock()
# m.


sim = SimulationEnvironment()

r = rospy.Rate(1)
while(not rospy.is_shutdown()):
    sim.step()
    print(sim.get_reward())
    state = sim.get_state()
    if state is not None:
        print(state.shape)
    sleep(2)







# world = np.zeros((20, 20))
#
# pose = np.array([0.4, 0.4, 0.0])
#
# map = Map((20, 20), values=world)
#
# map.set_robot_position(pose)
# map.set_value_cart(0.4, 0.4, 0)
#
# misc.imshow(map.values)
#
#
