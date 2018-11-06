MRDS_URL = "94.255.155.26:50000"

import httplib, json, time, sys
import matplotlib.pyplot as plt
from math import *

HEADERS = {"Content-type": "application/json", "Accept": "text/json"}

class UnexpectedResponse(Exception): pass

def post_speed(angular_speed, linear_speed):
    """Sends a speed command to the MRDS server"""
    mrds = httplib.HTTPConnection(MRDS_URL)
    params = json.dumps({'TargetAngularSpeed': angular_speed, 'TargetLinearSpeed': linear_speed})
    mrds.request('POST', '/lokarria/differentialdrive', params, HEADERS)
    response = mrds.getresponse()
    status = response.status
    if status == 204:
        return response
    else:
        raise UnexpectedResponse(response)

def get_pose():
    """Reads the current position and orientation from the MRDS"""
    mrds = httplib.HTTPConnection(MRDS_URL)
    mrds.request('GET', '/lokarria/localization')
    response = mrds.getresponse()
    if response.status == 200:
        pose_data = response.read()
        response.close()
        return json.loads(pose_data)
    else:
        raise UnexpectedResponse(response)

def bearing(q):
    return rotate(q, {'X': 1.0, 'Y': 0.0, "Z": 0.0})

def rotate(q, v):
    return vector(qmult(qmult(q, quaternion(v)), conjugate(q)))

def quaternion(v):
    q = v.copy()
    q['W'] = 0.0
    return q

def vector(q):
    v = {}
    v["X"] = q["X"]
    v["Y"] = q["Y"]
    v["Z"] = q["Z"]
    return v

def conjugate(q):
    qc = q.copy()
    qc["X"] = -q["X"]
    qc["Y"] = -q["Y"]
    qc["Z"] = -q["Z"]
    return qc

def qmult(q1, q2):
    q = {}
    q["W"] = q1["W"] * q2["W"] - q1["X"] * q2["X"] - q1["Y"] * q2["Y"] - q1["Z"] * q2["Z"]
    q["X"] = q1["W"] * q2["X"] + q1["X"] * q2["W"] + q1["Y"] * q2["Z"] - q1["Z"] * q2["Y"]
    q["Y"] = q1["W"] * q2["Y"] - q1["X"] * q2["Z"] + q1["Y"] * q2["W"] + q1["Z"] * q2["X"]
    q["Z"] = q1["W"] * q2["Z"] + q1["X"] * q2["Y"] - q1["Y"] * q2["X"] + q1["Z"] * q2["W"]
    return q

def get_heading():
    """Returns the XY Orientation as a bearing unit vector"""
    return bearing(get_pose()['Pose']['Orientation'])

def get_position():
    """Returns the XYZ position"""
    return get_pose()['Pose']['Position']

def pythagora_hypotenus(x, y):
    """Pythagoras theorem"""
    return sqrt((x ** 2) + (y ** 2))

def make_path():
    """Add all coordinates of the path to a stack"""
    stack = []
    with open(sys.argv[1]) as path_file:
        json_path = json.load(path_file)
        for i in range (len(json_path)):
            stack.append(json_path[i]['Pose']['Position'])
        stack.reverse()
        return stack

def get_point(path, pos, look_ahead, update_path = True):
    """Select the next goal point using the robot's position and a look ahead distance
    
    Args:
        path (array of points): The path that the robot must follow
        pos: The actual position of the robot
        look_ahead (float): The look ahead distance
        update_path (boolean): At true it deletes the past points, at false it doesn't

    Return the selected goal point
    """
    if path:
        for i in range(len(path)):
            point = path[len(path) - (1 if update_path else i)]
            dx = point['X'] - pos['X']
            dy = point['Y'] - pos['Y']

            dist = pythagora_hypotenus(dx, dy)

            if dist < look_ahead:
                if update_path:
                    path.pop()
            else:
                return point
    else:
        print ("Stack failed")

def pure_pursuit(robot_pos, point):
    """Compute the angular speed that the robot must have to follow the path 
    using the pure pursuit algorithm
    
    Args:
        robot_pos: the actual position of the robot
        point: the coordinates of the goal point

    Return the computed angular speed
    """
    dx = point['X'] - robot_pos['X']
    dy = point['Y'] - robot_pos['Y']
    dist = pythagora_hypotenus(dx, dy)

    robot_heading = get_heading()
    hx = robot_heading['X']
    hy = robot_heading['Y']
    robot_angle = atan2(hy, hx)
    point_angle = atan2(point['Y'] - robot_pos['Y'], point['X'] - robot_pos['X'])
    teta = point_angle - robot_angle
    delta_x = sin(teta) / dist

    return (2 * delta_x) / (dist ** 2)

def our_algo(robot_pos, point):
    """Compute the angular speed that the robot must have to follow the path 
    using our own algorithm
    
    Args:
        robot_pos: the actual position of the robot
        point: the coordinates of the goal point

    Return the computed angular speed
    """
    robot_heading = get_heading()
    hx = robot_heading['X']
    hy = robot_heading['Y']
    robot_angle = atan2(hy, hx)
    point_angle = atan2(point['Y'] - robot_pos['Y'], point['X'] - robot_pos['X'])

    teta = sin(point_angle - robot_angle)
    x = 0.3
    max_ang_speed = 3
    
    ang_speed = (max_ang_speed * teta) / x

    return min(max(ang_speed, -max_ang_speed), max_ang_speed)

def run_algo(algo, pos, goal_point):
    """Run the algorithm that the user choose
    
    Args:
        algo: The parameter value that the user gave
        pos: The actual position of the robot
        goal_point: The coordinates of the goal point

    Return the angular speed returned by the corresponding algorithm
    """
    ang_speed = 0
    if algo == 1:
        ang_speed = pure_pursuit(pos, goal_point)
    else:
        ang_speed = our_algo(pos, goal_point)

    return ang_speed

def compute_linear_speed(func, ang_speed):
    """Compute the linear speed adapted to the angular speed, using 
    the function that the user picked
    
    Args:
        func: The parameter value that the user gave
        ang_speed: The angular speed of the robot

    Return the computed linear speed
    """
    max_linear_speed = 1
    linear_speed = max_linear_speed

    if func == 2:
        linear_speed = min(max_linear_speed, 1 / log10(6 * abs(ang_speed) + 1))
    elif func == 3:
        linear_speed = min(max_linear_speed, log10(-(min(3, abs(ang_speed)) - 4.7)) + 0.5)
    elif func == 4:
        linear_speed = min(max_linear_speed, -0.2 * min(3, abs(ang_speed)) + 1.3)

    return linear_speed

if __name__ == '__main__':
    path = make_path()
    positions_x, positions_y, path_x, path_y = ([] for i in range(4))
    for point in path:
        path_x.append(point['X'])
        path_y.append(-point['Y'])

    angular_constant = 0.4
    look_ahead = 0.7

    algo, func, is_ahead = int(sys.argv[2]), int(sys.argv[3]), eval(sys.argv[4])
    if algo < 1 or algo > 2 or func < 1 or func > 4:
        print("Bad parameters, please refer to the 'How to run our program' section in our report.")
        sys.exit()

    start_time = time.time()
    while path:
        pos = get_position()
        positions_x.append(pos['X']) 
        positions_y.append(-pos['Y'])
        goal_point = get_point(path, pos, look_ahead)
        if goal_point:
            ang_speed = run_algo(algo, pos, goal_point)
            linear_speed = compute_linear_speed(func, ang_speed)
            if is_ahead:
                look_ahead = linear_speed

            response = post_speed(ang_speed * angular_constant, linear_speed)
            time.sleep(0.01)
    response = post_speed(0,0)

    end_time = time.time()
    run_time = end_time - start_time

    plt.plot(positions_y, positions_x, 'r', label="Robot")
    plt.plot(path_y, path_x, 'b', label="Path")
    plt.legend(loc='upper left')
    plt.show()

    print("end of run")
    print("The robot finished the path in:", run_time, "seconds")

