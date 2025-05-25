from TMMC_Wrapper import *
import rclpy
import time
import math

# ——————————————— Parameters ———————————————
challengeLevel = 3
is_SIM        = False
Debug         = False

# Safety‐brake
SAFE_STOP_DIST = 0.30   # m
BACKUP_SPEED   = -0.15  # m/s
# Drive
FIXED_SPEED    = 0.45   # m/s forward
# Stop sign
STOP_DURATION  = 3.0    # s
SIGN_COOLDOWN  = 5.0    # s
# AprilTag
TAG_TURN_THRESH = 0.75  # m
NUM_TAGS        = 4

# IMU turn parameters
ANGULAR_SPEED   = 0.5   # rad/s during rotation
YAW_TOLERANCE   = 5.0   # degrees

# ——————————————— Setup ———————————————
rclpy.init()
robot   = Robot(IS_SIM=is_SIM, DEBUG=Debug)
control = Control(robot)
camera  = Camera(robot)
imu     = IMU(robot)
lidar   = Lidar(robot)

# wrap send_cmd_vel for an immediate hard‐stop in a collision
_orig_send = control.send_cmd_vel
def safe_send(linear_x, angular_z):
    scan = robot.last_scan_msg
    if scan and linear_x > 0:
        dist, _ = lidar.detect_obstacle_in_cone(
            scan, distance=float("inf"), center=0.0, offset_angle=30.0
        )
        if dist < SAFE_STOP_DIST:
            linear_x = 0.0
    _orig_send(linear_x, angular_z)

control.send_cmd_vel = safe_send

def danger_close(threshold, half_fov=30.0):
    scan = lidar.checkScan()
    dist, _ = lidar.detect_obstacle_in_cone(
        scan, distance=threshold, center=0.0, offset_angle=half_fov
    )
    return dist != -1

def get_yaw_deg():
    # assume imu.get_euler() returns (roll, pitch, yaw) in radians
    _, _, yaw = imu.get_euler()
    return math.degrees(yaw) % 360

def angle_diff(target, current):
    # shortest signed difference target–current in degrees
    a = (target - current + 180) % 360 - 180
    return a

def rotate_to(target_deg):
    """Rotate in place until imu yaw is within YAW_TOLERANCE of target_deg."""
    while rclpy.ok():
        current = get_yaw_deg()
        err = angle_diff(target_deg, current)
        if abs(err) < YAW_TOLERANCE:
            break
        direction = 1 if err > 0 else -1
        _orig_send(0.0, direction * ANGULAR_SPEED)
        rclpy.spin_once(robot, timeout_sec=0.01)
    # stop rotating
    _orig_send(0.0, 0.0)

# ——————————————— Main Loop ———————————————
if challengeLevel == 3:
    control.stop_keyboard_control()
    tags_seen     = 0
    last_stop     = 0.0
    stop_sign_flag= False

    # kick off forward
    _orig_send(FIXED_SPEED, 0.0)

    try:
        while rclpy.ok() and tags_seen < NUM_TAGS:
            # 0) always refresh sensors
            rclpy.spin_once(robot, timeout_sec=0.01)

            # 1) collision avoidance: back up until safe
            if danger_close(SAFE_STOP_DIST):
                _orig_send(0.0, 0.0)
                time.sleep(0.01)
                # back up until no longer too close
                while danger_close(SAFE_STOP_DIST):
                    _orig_send(BACKUP_SPEED, 0.0)
                    time.sleep(0.1)
                    rclpy.spin_once(robot, timeout_sec=0.01)
                _orig_send(0.0, 0.0)
                # resume forward
                _orig_send(FIXED_SPEED, 0.0)
                continue

            # 2) stop‐sign detection
            img = camera.rosImg_to_cv2()
            now = time.time()
            if img is not None:
                seen, *_ = camera.ML_predict_stop_sign(img)
                if seen and (now - last_stop) >= SIGN_COOLDOWN:
                    last_stop = now
                    _orig_send(0.0, 0.0)
                    time.sleep(STOP_DURATION)
                    _orig_send(FIXED_SPEED, 0.0)
                    stop_sign_flag = True
                    continue

            # 3) AprilTag turns
            if img is not None:
                poses = camera.estimate_apriltag_pose(img)
                close = [p for p in poses if p[1] < TAG_TURN_THRESH]
                if close:
                    tags_seen += 1
                    _orig_send(0.0, 0.0)  # hard stop

                    if tags_seen < NUM_TAGS:
                        # decide turn angle
                        deg = 45 if stop_sign_flag else 90
                        # compute target yaw
                        curr_yaw = get_yaw_deg()
                        target_yaw = (curr_yaw + deg) % 360
                        rotate_to(target_yaw)
                        stop_sign_flag = False
                        _orig_send(FIXED_SPEED, 0.0)
                    else:
                        # final stop at last tag
                        print("Loop A complete — reached tag #4, stopping.")
                        break
                    continue

            # 4) default forward
            _orig_send(FIXED_SPEED, 0.0)
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Keyboard interrupt — shutting down.")

    finally:
        control.stop_keyboard_control()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()