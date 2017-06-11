import numpy as np
import Image
from Transformations import tf_robot_to_world, polar_to_cart

from sensor_msgs.msg import LaserScan

class Map:
    original_values = None
    values = None
    resolution = None
    origin = None
    size = None
    robot_pose = None
    goal = None
    scan = None
    
    def __init__(self, size, values=None, resolution=0.05, origin=(0.0, 0.0)):
        self.resolution = resolution
        self.origin = np.array(origin)
        self.size = size
        
        if values is None:
            self.values = np.zeros(size)
        else:
            self.values = np.asarray(values, int).reshape(size).transpose()
            # self.values[self.values<128] = 0
            self.values[self.values>0] = 254
            
        self.original_values = self.values.copy()

    def get_cell(self, x, y):
        '''
        Returns the cell index in x and y direction for the given x/y coordinates in the world frame.
        :param x: The x coordinate in world frame.
        :param y: The y coordinate in world frame.
        :return: (x, y) Tuple of cell indexes in x and y direction within the map
        '''
        x = np.floor(x / self.resolution) + np.floor(self.origin[0] / self.resolution)
        y = np.floor(y / self.resolution) + np.floor(self.origin[1] / self.resolution)
        
        if x >= self.size[0] or y >= self.size[1]:
            print('Error, cell out of map bounds!')
            # ToDo: Maybe increase map size at this point
            return False
    
        return int(x), int(y)

    def set_value_cart(self, point, value):
        '''
        Sets the value of a cell at position x and y in euler coordinates in the world frame.
        :param x: The x coordinate in world frame.
        :param y: The y coordinate in world frame.
        :param value: The value to set.
        :return:
        '''
        cell_index = self.get_cell(point[0], point[1])
        if not cell_index:
            return
        self.values[cell_index] = value
        
    def set_value_index(self, x, y, value):
        '''
        Sets the value of a cell at position x and y in euler coordinates in the world frame.
        :param x: The x coordinate in world frame.
        :param y: The y coordinate in world frame.
        :param value: The value to set.
        :return:
        '''
        if x >= self.size[0] or y >= self.size[1]:
            print('Error, cell out of map bounds!')
            # ToDo: Maybe increase map size at this point
            print(x, y)
            return False
        
        self.values[x, y] = value
        return True

    def set_robot_position(self, robot_pose):
        self.robot_pose = robot_pose
        
    def set_goal(self, goal):
        self.goal = goal
        
    def set_scan(self, scan):
        self.scan = scan

    def draw_robot_position(self, robot_pose, value=160):
        print 'Drawing the robots positional triangle on the map...'
        
        # self.values = self.original_values.copy()
        
        # The points of the triangle surrounding the robots position in steps
        triangle = np.array([[-3, -3],
                             [ 0,  6],
                             [ 3, -3]])
        # triangle = np.array([[-2, -2],
        #                      [ -2,  2],
        #                      [ 2, 2],
        #                      [ 2, -2]])
        
        robot_points = np.zeros(triangle.shape)
        robot_point_indexes = np.zeros(triangle.shape, int)
        robot_pose[2] += -np.pi*0.5
        
        for i in range(triangle.shape[0]):
            # Get points from triangle in map coordinate frame
            robot_points[i] = tf_robot_to_world(robot_pose, triangle[i]*self.resolution)
            # Get cell indexes for map coordinates
            robot_point_indexes[i] = self.get_cell(robot_points[i, 0], robot_points[i, 1])
        
        # Get all points along the edges of the triangle
        triangle_points = []
        for i in range(robot_point_indexes.shape[0]-1):
            for k in range(i+1, robot_point_indexes.shape[0]):
                for c in self.covered_cells(robot_point_indexes[i], robot_point_indexes[k]):
                    triangle_points.append(c)
                    
        triangle_points.append(robot_point_indexes[robot_point_indexes.shape[0]-1])
                    
        # Connect all edge points using raytracing
        for i in range(len(triangle_points)-1):
            for k in range(i+1, len(triangle_points)):
                for c in self.covered_cells(triangle_points[i], triangle_points[k]):
                    self.set_value_index(c[0], c[1], value)
                    # print(c)

        self.set_value_index(robot_point_indexes[robot_point_indexes.shape[0]-1, 0], robot_point_indexes[robot_point_indexes.shape[0]-1, 1], value)

    def draw_goal_position(self, goal, value=80):
        print 'Drawing the goal position on the map...'
    
        # self.values = self.original_values.copy()
    
        # The points of the triangle surrounding the robots position in steps
        cross = np.array([[-3, -3],
                          [-3,  3],
                          [ 3,  3],
                          [ 3, -3]], float)
        cross *= self.resolution
        cross += goal
        
        cross_indexes = np.zeros(cross.shape, int)
        
        for i in range(cross.shape[0]):
            cross_indexes[i] = self.get_cell(cross[i, 0], cross[i, 1])
    
        # Connect the first two corners
        for c in self.covered_cells(cross_indexes[0], cross_indexes[2]):
            self.set_value_index(c[0], c[1], value)
        self.set_value_index(cross_indexes[2, 0], cross_indexes[2, 1], value)
        # Connect other two corners
        for c in self.covered_cells(cross_indexes[1], cross_indexes[3]):
            self.set_value_index(c[0], c[1], value)
        self.set_value_index(cross_indexes[3, 0], cross_indexes[3, 1], value)

    def draw_robot_scan(self, robot_pose, scan, value=254):
        print 'Drawing the robots laserscan on the map...'
    
        # self.values = self.original_values.copy()
        
        robot_pose[2] += np.pi*0.5
        
        # Loop through all scan points as ranges per angle increment
        angle = scan.angle_min
        for i in range(len(scan.ranges)):
            # Check if range is smaller than max range
            if scan.ranges[i] >= scan.range_max or scan.ranges[i] <= scan.range_min:
                continue
            # Convert point to cart coordinates
            point = polar_to_cart(scan.ranges[i], angle)
            # Transform point into map frame
            point = tf_robot_to_world(robot_pose, point)
            # Check for nan values
            if np.isnan(point).any():
                continue
            # Draw point in map
            self.set_value_cart(point, value)
            # Increment angle
            angle += scan.angle_increment

    def draw_all(self):
        
        if self.robot_pose is None or self.goal is None or self.scan is None:
            print('Error! Not all attributes have been set!')
            print(self.robot_pose is None)
            print(self.goal is None)
            print(self.scan is None)

            return False
        
        self.values = self.original_values.copy()
        
        self.draw_goal_position(self.goal)
        self.draw_robot_position(self.robot_pose)
        self.draw_robot_scan(self.robot_pose, self.scan)
        
        return True

    def covered_cells(self, start, end):
        """Cells covered by a ray from the start cell to the end cell.

        Arguments:
        start -- (x,y) position of the start cell
        end -- (x,y) position of the end cell
        """
    
        # We'll need the lengths of the ranges later on.
        x_range = end[0] - start[0]
        y_range = end[1] - start[1]
    
        # If the start and the end are the same point, then we don't do
        # anything.
        if x_range == 0 and y_range == 0:
            return
            yield
    
        # Step through x or y?  Pick the one with the longer absolute
        # range.
        if abs(x_range) > abs(y_range):
            y_step = float(y_range) / abs(float(x_range))
            y = float(start[1])
            for x in xrange(start[0], end[0], np.sign(x_range)):
                yield ((x, int(round(y))))
                y += y_step
        else:
            x_step = float(x_range) / abs(float(y_range))
            x = float(start[0])
            for y in xrange(start[1], end[1], np.sign(y_range)):
                yield ((int(round(x)), y))
                x += x_step

    def save_map_to_img(self, range=15, file_name='grid_map'):
        '''
        Creates a new image and saves the given map data in it.
        :param size: The size of the image in Pixels.
        :param data: The map-data as a numpy-array.
        :param range: The data range of the map values.
        :param file_name: The file-name for the saved image.
        :return: None
        '''
        # Create new image
        img = Image.new('L', self.size)
        # Save map data to image
        img.putdata(self.values.flatten(), 1, 0)
        # Save image to disk
        img.save('/home/robin/catkin_ws/src/int_agents_project/data/' + file_name + '.png')
        # img.show()
    
        print('Map saved successfully.')
        
    # def check_area_for_obstacles(self, x, y, range):
    #
    #     for x_i in range(range):
            