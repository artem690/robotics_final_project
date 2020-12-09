"""robot controller."""
import math
import copy
from controller import Robot, Motor, DistanceSensor
import supervisor
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import sys


class Node:
    def __init__(self, pt, parent=None, path_from_parent=[]):
        self.point = pt # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = path_from_parent # List containing from point and to_point for visualizing


# create the Robot instance.
supervisor.init_supervisor()
robot = supervisor.supervisor

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

#positions of obstacles x y 
TARGETS = supervisor.supervisor_get_targets()

# Robot Pose Values
pose_x = 0
pose_y = 0
pose_theta = 0
left_wheel_direction = 0
right_wheel_direction = 0

EPUCK_MAX_WHEEL_SPEED = 0.12880519 # m/s
EPUCK_AXLE_DIAMETER = 0.053 # ePuck's wheels are 53mm apart.
EPUCK_WHEEL_RADIUS = 0.0205 # ePuck's wheels are 0.041m in diameter.

# Initialize Motors
leftMotor = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

MAX_VEL_REDUCTION = 0.25

MAP_BOUNDS = np.array([[0,1.5],[0,1.5]])
OBSTACLES = np.array(supervisor.supervisor_get_obstacle_positions())
OBSTACLES = list(map(lambda x: [x[0]+0.25,x[1],x[2]], OBSTACLES))
LINE_SEGMENTS = []

#distance sensor initialization
ps = []
psNames = ['ps0', 'ps1', 'ps2', 'ps3','ps4', 'ps5', 'ps6', 'ps7']
for i in range(8):
    ps.append(robot.getDistanceSensor(psNames[i]))
    ps[i].enable(timestep)

def update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed):
  
    global pose_x, pose_y, pose_theta, EPUCK_MAX_WHEEL_SPEED, EPUCK_AXLE_DIAMETER
    pose_theta += (right_wheel_direction - left_wheel_direction) * time_elapsed * EPUCK_MAX_WHEEL_SPEED / EPUCK_AXLE_DIAMETER
    pose_x += math.cos(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction)/2.
    pose_y += math.sin(pose_theta) * time_elapsed * EPUCK_MAX_WHEEL_SPEED * (left_wheel_direction + right_wheel_direction)/2.
    pose_theta = get_bounded_theta(pose_theta)
    
def get_bounded_theta(theta):
    
    while theta > math.pi: theta -= 2.*math.pi
    while theta < -math.pi: theta += 2.*math.pi
    return theta

def get_wheel_speeds(target_pose):
    
    global pose_x, pose_y, pose_theta, left_wheel_direction, right_wheel_direction

    pose_x, pose_y, pose_theta = csci3302_lab5_supervisor.supervisor_get_robot_pose()


    bearing_error = math.atan2( (target_pose[1] - pose_y), (target_pose[0] - pose_x) ) - pose_theta
    distance_error = np.linalg.norm(target_pose[:2] - np.array([pose_x,pose_y]))
    heading_error = target_pose[2] -  pose_theta

    BEAR_THRESHOLD = 0.06
    DIST_THRESHOLD = 0.03
    dT_gain = theta_gain
    dX_gain = distance_gain
    if distance_error > DIST_THRESHOLD:
        dTheta = bearing_error
        if abs(bearing_error) > BEAR_THRESHOLD:
            dX_gain = 0
    else:
        dTheta = heading_error
        dX_gain = 0

    dTheta *= dT_gain
    dX = dX_gain * min(3.14159, distance_error)

    phi_l = (dX - (dTheta*EPUCK_AXLE_DIAMETER/2.)) / EPUCK_WHEEL_RADIUS
    phi_r = (dX + (dTheta*EPUCK_AXLE_DIAMETER/2.)) / EPUCK_WHEEL_RADIUS

    left_speed_pct = 0
    right_speed_pct = 0
    
    wheel_rotation_normalizer = max(abs(phi_l), abs(phi_r))
    left_speed_pct = (phi_l) / wheel_rotation_normalizer
    right_speed_pct = (phi_r) / wheel_rotation_normalizer
    
    if distance_error < 0.05 and abs(heading_error) < 0.05:    
        left_speed_pct = 0
        right_speed_pct = 0
        
    left_wheel_direction = left_speed_pct * MAX_VEL_REDUCTION
    phi_l_pct = left_speed_pct * MAX_VEL_REDUCTION * leftMotor.getMaxVelocity()

    right_wheel_direction = right_speed_pct * MAX_VEL_REDUCTION
    phi_r_pct = right_speed_pct * MAX_VEL_REDUCTION * rightMotor.getMaxVelocity()


    # print("Current pose: [%5f, %5f, %5f]\t\t Target pose: [%5f, %5f, %5f]\t\t %5f %5f %5f\t\t  %3f %3f" % (pose_x, pose_y, pose_theta, target_pose[0], target_pose[1], target_pose[2], bearing_error, distance_error, get_bounded_theta(heading_error), left_wheel_direction, right_wheel_direction))
   
    return phi_l_pct, phi_r_pct
    

def visualize_2D_graph(state_bounds, line_segments, nodes, goals, paths, filename=None):
    global TARGETS
    fig = plt.figure()
    plt.xlim(state_bounds[0,0], state_bounds[0,1])
    plt.ylim(state_bounds[1,0], state_bounds[1,1])
    t = 1.5
    goal_point=None
    
    for targ in TARGETS:
        x,y = targ[0],targ[1]
        plt.plot(x, t-y, 'kX', markersize=10)
        
    for seg in line_segments:
        [x1,y1], [x2,y2] = seg
        plt.plot([x1,x2], [t-y1,t-y2], marker = 'o')
    goal_node = None
    for node in nodes:
        if node.parent is not None:
            node_path = np.array(node.path_from_parent)
            plt.plot(node_path[:,0], t - node_path[:,1], '-b')
        if goal_point is not None and np.linalg.norm(node.point - np.array(goal_point)) <= 1e-5:
            goal_node = node
            plt.plot(node.point[0], t - node.point[1], 'k^')
        else:
            plt.plot(node.point[0], t - node.point[1], 'ro')

    plt.plot(nodes[0].point[0], t - nodes[0].point[1], 'ko')
    
    if goals is not None:
        for goal_node in goals:
            cur_node = goal_node
            while cur_node is not None:
                if cur_node.parent is not None:
                    node_path = np.array(cur_node.path_from_parent)
                    plt.plot(node_path[:,0], t - node_path[:,1], '--y')
                
                if cur_node in paths:
                    cur_node = paths[cur_node]
                else:
                    cur_node = None

    if goal_point is not None:
        plt.plot(goal_point[0], goal_point[1], 'gx')


    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()    
    
    
def obsTransform(obstacles):
    global LINE_SEGMENTS
    
    for obs in obstacles:
          x,y = obs[0]
          l = obs[1]
          theta = obs[2]
        
          lower_coord = (x - l * np.cos(theta) / 2, y - l * np.sin(theta) / 2)
          upper_coord = (x + l * np.cos(theta) / 2, y + l * np.sin(theta) / 2)
        
          LINE_SEGMENTS.append(np.array([lower_coord, upper_coord]))
    return
        
        
def steer(from_point, to_point, delta_q):
    dist = np.linalg.norm(to_point - from_point)

    if dist > delta_q:
        # formula to get point on edge of circle from -> https://math.stackexchange.com/a/127615
        to_point = from_point + delta_q * (to_point - from_point) / np.linalg.norm(to_point - from_point)

    return (from_point, to_point)
       

def get_nearest_vertex(node_list, q_point):
    filtered_node_list = [x.point for x in node_list]
    index = np.argmin(np.linalg.norm(filtered_node_list-np.array(q_point), axis=1))
    return node_list[index]

# bound check and if crosses any line segments (mazeblocks)
def state_is_valid(from_point, to_point):
    for dim in range(MAP_BOUNDS.shape[0]):
        if to_point[dim] < MAP_BOUNDS[dim][0]: return False
        if to_point[dim] >= MAP_BOUNDS[dim][1]: return False
    for lines in LINE_SEGMENTS:
        t, s = np.linalg.solve(np.array([to_point-from_point, lines[0]-lines[1]]).T, lines[0]-from_point)
        x, y = (1-t)*from_point + t*to_point
        x1, x2, y1, y2 = min(from_point[0], to_point[0]), max(from_point[0], to_point[0]), \
                         min(from_point[1], to_point[1]), max(from_point[1], to_point[1])
        lx1, lx2, ly1, ly2 = min(lines[0][0], lines[1][0]), max(lines[0][0], lines[1][0]), \
                             min(lines[0][1], lines[1][1]), max(lines[0][1], lines[1][1])
        
        # look for intersection on both line segments
        if x >= x1 and x <= x2 and y >= y1 and y <= y2 and \
           x >= lx1 and x <= lx2 and y >= ly1 and y <= ly2:
            return False
    return True
    
    
def get_random_vertex(bounds, obstacles):
    vertex = None
    while vertex is None: # Get starting vertex
        vertex = np.random.rand(bounds.shape[0]) * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
    return vertex    
    

def build_rrt(state_bounds, obstacles, state_is_valid, starting_point, goal_point, k, delta_q):
    node_list = []
    node_list.append(starting_point) # Add Node at starting point with no parent

    for i in range(k):
        q_rand = get_random_vertex(state_bounds, obstacles)
        q_near = get_nearest_vertex(node_list, q_rand)
        from_point, to_point = steer(q_near.point, q_rand, delta_q)
        
        has_obstacle = False
        if not state_is_valid(from_point, to_point):
            has_obstacle = True
            
        # add if path is clear
        if not has_obstacle:
            node_list.append(
                Node(to_point, parent=q_near, path_from_parent=[from_point, to_point])
            )

    return node_list
    

def get_path(node_list, start_node):
    global TARGETS
    goals = TARGETS[:]
    visited = {start_node:None}

    for ix,goal in enumerate(goals):
        goal_parent = get_valid_connect(node_list, goal)
        
        if goal_parent != -1:
            goal_node = Node(goal, parent=goal_parent, path_from_parent=[goal, goal_parent.point])
            node_list.append(goal_node)
            goals[ix] = goal_node
        else:
            goals[ix] = None

    goals = [g for g in goals if g] # remove None
    for goal in goals:
        cur_node = goal
        
        while cur_node:
            found = False
            visited[cur_node] = cur_node.parent
            cur_node = cur_node.parent
            
    return visited, goals
            
        
def get_valid_connect(node_list, q_point):
    valid_connects = []
    valid_indices = []
            
    for i,node in enumerate(node_list):
        has_obstacle = False
        if not state_is_valid(np.array(node.point), np.array(q_point)):
            has_obstacle = True
        
        if not has_obstacle:
            valid_connects.append(node.point)
            valid_indices.append(i)
    
    if len(valid_connects) > 0:
        # retrieve minimum distance index
        index = np.argmin(np.linalg.norm(np.array(valid_connects)-np.array(q_point), axis=1))
        
        return node_list[valid_indices[index]]
    else:
        return -1
    
def get_sensor_data():
    global ps

    psValues = []
    for i in range(8):
        psValues.append(ps[i].getValue())
    return psValues



def main():
    global OBSTACLES, LINE_SEGMENTS
    # You should insert a getDevice-like function in order to get the
    # instance of a device of the robot. Something like:
    #  motor = robot.getMotor('motorname')
    #  ds = robot.getDistanceSensor('dsname')
    #  ds.enable(timestep)2
    K = 500 # adjustable
    start_pose = supervisor.supervisor_get_robot_pose()[:2]
    start_pose+=.25
    starting_point = Node(start_pose[:2], parent=None)
    obsTransform(OBSTACLES)
    node_list = build_rrt(MAP_BOUNDS, OBSTACLES, state_is_valid, starting_point, None, K, np.linalg.norm(MAP_BOUNDS/20.))
    # pose_x, pose_y, pose_theta = start_pose
    paths, goals = get_path(node_list, starting_point)
    
    visualize_2D_graph(MAP_BOUNDS, LINE_SEGMENTS, node_list, goals, paths, 'rrt_maze_run.png')
    # print(goals[-1].point)
    # g = goals[-1]
    # while path[g] != None:
        # print(path[g].point)
        # g = path[g]
    pose_x, pose_y, pose_theta = start_pose
    last_odometry_update_time = None
    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        if last_odometry_update_time is None:
            last_odometry_update_time = robot.getTime()
        time_elapsed = robot.getTime() - last_odometry_update_time
        update_odometry(left_wheel_direction, right_wheel_direction, time_elapsed)
        last_odometry_update_time = robot.getTime()
        # Read the sensors:
        # Enter here functions to read sensor data, like:
        dist_val=get_sensor_data()
    
        # Process sensor data here.
    
        # Enter here functions to send actuator commands, like:
        #  motor.setPosition(10.0)
        break
    
    # Enter here exit cleanup code.  
    print("BYE")  
    


if __name__ == "__main__":
    main()
