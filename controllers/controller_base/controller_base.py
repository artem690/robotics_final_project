"""robot controller."""
import math
import copy
from controller import Robot, Motor, DistanceSensor
import supervisor
import numpy as np
from collections import defaultdict

# create the Robot instance.
supervisor.init_supervisor()
robot = supervisor.supervisor

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
obstacle_duck1 = (0.07, 0.01, 0.95)
obstacle_duck2 = (-0.150001, 0.01, 0.58)
obstacle_duck3 = (0.79, 0.01, -0.03)
obstacle_soda = (-0.18, 0.06, -0.17)

def hasObstacle():
    pass
    
def buildRRT():
    pass

def main():
    # You should insert a getDevice-like function in order to get the
    # instance of a device of the robot. Something like:
    #  motor = robot.getMotor('motorname')
    #  ds = robot.getDistanceSensor('dsname')
    #  ds.enable(timestep)
    print("well?")
    print(supervisor.supervisor_get_obstacle_positions())
    print("well?")
    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while robot.step(timestep) != -1:
        # Read the sensors:
        # Enter here functions to read sensor data, like:
        #  val = ds.getValue()
    
        # Process sensor data here.
    
        # Enter here functions to send actuator commands, like:
        #  motor.setPosition(10.0)
        pass
    
    # Enter here exit cleanup code.    
    


if __name__ == "__main__":
    main()