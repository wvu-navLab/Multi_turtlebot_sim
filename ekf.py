#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import numpy as np
from scipy.spatial.transform import Rotation as R
from multi_turtlebot_sim.msg import RobotState, RobotCovariance


def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def construct_F(CbnPlus, accel):
    F = np.zeros((15, 15))

    # Populate F matrix based on the given instructions
    F[3:6, 0:3] = skew_symmetric(-CbnPlus @ accel)
    F[0:3, 12:15] = CbnPlus
    F[3:6, 9:12] = CbnPlus
    F[6:9, 3:6] = np.eye(3)
    
    return F

def calc_Q(dt, CbnPlus, accel):
    gg = 9.80665
    
    # Convert biases and noise to SI units
    sig_gyro_inRun = 0.2 * np.pi / 180 / 3600  # rad/s
    sig_ARW = 0.3 * np.pi / 180 * np.sqrt(3600) / 3600  # rad
    sig_accel_inRun = 3e-3 * gg  # m/s^2
    sig_VRW = 0.023 * np.sqrt(3600) / 3600  # m/s
    
    # Calculate power spectral densities
    Srg = (sig_ARW ** 2) * dt
    Sra = (sig_VRW ** 2) * dt
    Sbad = (sig_accel_inRun ** 2) / dt
    Sbgd = (sig_gyro_inRun ** 2) / dt
    
    # Initialize sub-matrices
    F21 = -skew_symmetric(np.dot(CbnPlus, accel))
    
    # Identity matrix for transformation in XYZ frame
    T_rn_p = np.eye(3)
    
    Q11 = (Srg * dt + (1.0 / 3.0) * Sbgd * (dt ** 3)) * np.eye(3)
    Q21 = ((1.0 / 2.0) * Srg * (dt ** 2) + (1.0 / 4.0) * Sbgd * (dt ** 4)) * F21
    Q12 = Q21.T
    Q31 = ((1.0 / 3.0) * Srg * (dt ** 3) + (1.0 / 5.0) * Sbgd * (dt ** 5)) * np.dot(T_rn_p, F21)
    Q13 = Q31.T
    Q14 = np.zeros((3, 3))
    Q15 = (1.0 / 2.0) * Sbgd * (dt ** 2) * CbnPlus
    Q22 = (Sra * dt + (1.0 / 3.0) * Sbad * (dt ** 3)) * np.eye(3) + ((1.0 / 3.0) * Srg * (dt ** 3) + (1.0 / 5.0) * Sbgd * (dt ** 5)) * np.dot(F21, F21.T)
    Q32 = ((1.0 / 2.0) * Sra * (dt ** 2) + (1.0 / 4.0) * Sbad * (dt ** 4)) * T_rn_p + ((1.0 / 4.0) * Srg * (dt ** 4) + (1.0 / 6.0) * Sbgd * (dt ** 6)) * np.dot(T_rn_p, np.dot(F21, F21.T))
    Q23 = Q32.T
    Q24 = (1.0 / 2.0) * Sbad * (dt ** 2) * CbnPlus
    Q25 = (1.0 / 3.0) * Sbgd * (dt ** 3) * np.dot(F21, CbnPlus)
    Q33 = ((1.0 / 3.0) * Sra * (dt ** 3) + (1.0 / 5.0) * Sbad * (dt ** 5)) * np.dot(T_rn_p, T_rn_p) + ((1.0 / 5.0) * Srg * (dt ** 5) + (1.0 / 7.0) * Sbgd * (dt ** 7)) * np.dot(T_rn_p, np.dot(F21, np.dot(F21.T, T_rn_p)))
    Q34 = (1.0 / 3.0) * Sbad * (dt ** 3) * np.dot(T_rn_p, CbnPlus)
    Q35 = (1.0 / 4.0) * Sbgd * (dt ** 4) * np.dot(T_rn_p, np.dot(F21, CbnPlus))
    Q41 = np.zeros((3, 3))
    Q42 = (1.0 / 2.0) * Sbad * (dt ** 2) * CbnPlus.T
    Q43 = Q34.T
    Q44 = Sbad * dt * np.eye(3)
    Q45 = np.zeros((3, 3))
    Q51 = (1.0 / 2.0) * Sbgd * (dt ** 2) * CbnPlus.T
    Q52 = (1.0 / 3.0) * Sbgd * (dt ** 3) * np.dot(F21.T, CbnPlus.T)
    Q53 = Q35.T
    Q54 = np.zeros((3, 3))
    Q55 = Sbgd * dt * np.eye(3)
    
    # Assemble the full Q matrix
    Q_block = np.block([
        [Q11, Q12, Q13, Q14, Q15],
        [Q21, Q22, Q23, Q24, Q25],
        [Q31, Q32, Q33, Q34, Q35],
        [Q41, Q42, Q43, Q44, Q45],
        [Q51, Q52, Q53, Q54, Q55]
    ])
    Q = np.block([
        [Q_block, np.zeros((15, 15)), np.zeros((15, 15))],
        [np.zeros((15, 15)), Q_block, np.zeros((15, 15))],
        [np.zeros((15, 15)), np.zeros((15, 15)), Q_block]
    ])
    return Q

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')
        self.initialize_filter()
        self.create_subscriptions()
        self.create_publishers()
        self.timer = self.create_timer(0.1, self.timer_callback)

    def initialize_filter(self):
        # State vector: [attitude, velocity, position, acc bias, gyro bias] for each robot
        self.nominal_state = np.zeros(45)  # 15 states per robot, 3 robots
        self.error_state = np.zeros(45)    # 15 error states per robot, 3 robots
        self.covariance = np.eye(45) * 0.1
        self.initialized = [False, False, False]
        self.f_ib_b = [np.zeros(3), np.zeros(3), np.zeros(3)]
        self.gyro_world = [np.zeros(3), np.zeros(3), np.zeros(3)]

    def create_subscriptions(self):
        self.create_subscription(Imu, '/robot1/imu', self.imu_callback_1, 10)
        self.create_subscription(Imu, '/robot2/imu', self.imu_callback_2, 10)
        self.create_subscription(Imu, '/robot3/imu', self.imu_callback_3, 10)
        self.create_subscription(Float64MultiArray, '/robot1_2/uwb', self.uwb_callback_1_2, 10)
        self.create_subscription(Float64MultiArray, '/robot1_3/uwb', self.uwb_callback_1_3, 10)
        self.create_subscription(Float64MultiArray, '/robot2_3/uwb', self.uwb_callback_2_3, 10)
        self.create_subscription(Odometry, '/robot1/odom', self.odom_callback_1, 10)
        self.create_subscription(Odometry, '/robot2/odom', self.odom_callback_2, 10)
        self.create_subscription(Odometry, '/robot3/odom', self.odom_callback_3, 10)

    def create_publishers(self):
        self.state_pub_1 = self.create_publisher(RobotState, '/robot1/state', 10)
        self.covariance_pub_1 = self.create_publisher(RobotCovariance, '/robot1/covariance', 10)
        self.state_pub_2 = self.create_publisher(RobotState, '/robot2/state', 10)
        self.covariance_pub_2 = self.create_publisher(RobotCovariance, '/robot2/covariance', 10)
        self.state_pub_3 = self.create_publisher(RobotState, '/robot3/state', 10)
        self.covariance_pub_3 = self.create_publisher(RobotCovariance, '/robot3/covariance', 10)

    def predict(self, dt):
        g = np.array([0, 0, -9.81])  # gravity vector

        for i in range(3):
            idx = i * 15
            roll, pitch, yaw = self.nominal_state[idx], self.nominal_state[idx+1], self.nominal_state[idx+2]
            Cbn = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
            gyro = self.gyro_world[i] - self.nominal_state[idx+12:idx+15]
            accel = self.f_ib_b[i] #- self.nominal_state[idx+9:idx+12]
            print("accel", accel)

            #print("Cbn", Cbn)

            # Update CbnPlus
            CbnPlus = Cbn @ (np.eye(3) + skew_symmetric(gyro) * dt)
            #CbnPlus = np.eye(3)

            # Velocity prediction
            v_minus = self.nominal_state[idx+3:idx+6]
            v_plus = v_minus + ((CbnPlus @ accel) + g) * dt
            self.nominal_state[idx+3:idx+6] = v_plus

            # Position prediction
            r_minus = self.nominal_state[idx+6:idx+9]
            r_plus = r_minus - (dt / 2) * (v_minus + v_plus)
            self.nominal_state[idx+6:idx+9] = r_plus



            # Construct F matrix
            F = construct_F(CbnPlus, accel)

            # Calculate Q matrix
            Q = calc_Q(dt, CbnPlus, accel)

            #print("Q[3:6, 3:6]:\n", Q[12:15, 12:15])

            F1_block = np.eye(15) + F * dt

            F1 = np.block([
                [F1_block, np.zeros((15, 15)), np.zeros((15, 15))],
            [np.zeros((15, 15)), F1_block, np.zeros((15, 15))],
            [np.zeros((15, 15)), np.zeros((15, 15)), F1_block]
            ])

            self.error_state = F1 @ self.error_state

            # Propagate error state covariance
            self.covariance = F1 @ self.covariance @ F1.T + Q

    def update(self, z, R, sensor_type, robot_idx1, robot_idx2=None, zero_velocity = False):
        if sensor_type == 'uwb':
            idx1 = robot_idx1 * 15
            idx2 = robot_idx2 * 15
            H = np.zeros((1, 45))
            # Predicted measurement using current state estimate
            px1, py1, pz1 = self.nominal_state[idx1+6], self.nominal_state[idx1+7], self.nominal_state[idx1+8]
            px2, py2, pz2 = self.nominal_state[idx2+6], self.nominal_state[idx2+7], self.nominal_state[idx2+8]
            predicted_distance = np.sqrt((px1 - px2)**2 + (py1 - py2)**2 + (pz1 - pz2)**2)
            # Partial derivatives for the measurement model
            H[0, idx1+6] = (px1 - px2) / predicted_distance
            H[0, idx1+7] = (py1 - py2) / predicted_distance
            H[0, idx1+8] = (pz1 - pz2) / predicted_distance
            H[0, idx2+6] = (px2 - px1) / predicted_distance
            H[0, idx2+7] = (py2 - py1) / predicted_distance
            H[0, idx2+8] = (pz2 - pz1) / predicted_distance

            # Measurement residual
            y = z - H @ self.nominal_state

        elif sensor_type == 'odom':
            idx = robot_idx1 * 15
            H = np.zeros((3, 45))  
            H[0, idx+3] = -1  # vx
            H[1, idx+4] = -1  # vy
            H[2, idx+5] = -1  # vz
            # Measurement residual for velocity
            y = z - H @ self.nominal_state # Only compare velocity

        # Zero Velocity Update (ZVU)
        '''elif zero_velocity:
            idx = robot_idx1 * 15
            H = np.zeros((6, 45))  
            H[0:3, idx+3:idx+6] = -np.eye(3)  # H[0:3, 3:6]
            H[3:6, idx+12:idx+15] = -np.eye(3)  # H[3:6, 12:15]

            # Measurement residual for zero velocity
            y = -H @ self.nominal_state[idx+3:idx+9]  # Assume measurement is zero

            # Adjust R to match the ZVU size
            R = np.eye(6) * 0.01 '''

           

        # Measurement covariance
        S = H @ self.covariance @ H.T + R
        
        # Ensure S is positive definite before inverting
        if np.all(np.linalg.eigvals(S) > 0):
            K = self.covariance @ H.T @ np.linalg.inv(S)
            # Update the error state and covariance matrix
            delta_error_state = K @ y
            #print(delta_error_state)
            I = np.eye(45)
            self.covariance = (I - K @ H) @ self.covariance
        else:
            self.get_logger().warn('Measurement covariance matrix is not positive definite. Skipping update.')
            return

        # Correct the nominal state with the error state
        self.nominal_state += delta_error_state
        # Reset the error state
        self.error_state = np.zeros(45)

    def imu_callback_1(self, msg):
        self.process_imu(msg, 0)

    def imu_callback_2(self, msg):
        self.process_imu(msg, 1)

    def imu_callback_3(self, msg):
        self.process_imu(msg, 2)

    def process_imu(self, msg, robot_index):
        imu_accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        imu_gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]) * np.pi / 180 # rad/sec
        
        # Store measurements for use in predict function
        #imu_accel[2] = 9.8
        self.f_ib_b[robot_index] = imu_accel

        self.gyro_world[robot_index] = imu_gyro

    def uwb_callback_1_2(self, msg):
        self.update(msg.data[0], np.array([[0.04]]), 'uwb', 0, 1)

    def uwb_callback_1_3(self, msg):
        self.update(msg.data[0], np.array([[0.04]]), 'uwb', 0, 2)

    def uwb_callback_2_3(self, msg):
        self.update(msg.data[0], np.array([[0.04]]), 'uwb', 1, 2)

    def odom_callback_1(self, msg):
        self.process_odom(msg, 0)

    def odom_callback_2(self, msg):
        self.process_odom(msg, 1)

    def odom_callback_3(self, msg):
        self.process_odom(msg, 2)

    def process_odom(self, msg, robot_index):
        if not self.initialized[robot_index]:
            self.initialized[robot_index] = True

        odom_velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        R_mat = np.diag([0.04] * 3)  # Only 3x3 matrix for velocity noise
        self.update(odom_velocity, R_mat, 'odom', robot_index)


    def timer_callback(self):
        dt = 0.1  # Time step
        self.predict(dt)
        self.publish_state_and_covariance()

    def publish_state_and_covariance(self):
        robot_state_msgs = []
        robot_covariance_msgs = []

        for i in range(3):
            idx = i * 15
            state_msg = RobotState()
            state_msg.state = self.nominal_state[idx:idx+15].tolist()

            covariance_msg = RobotCovariance()
            covariance_msg.covariance = self.covariance[idx:idx+15, idx:idx+15].flatten().tolist()

            robot_state_msgs.append(state_msg)
            robot_covariance_msgs.append(covariance_msg)

        self.state_pub_1.publish(robot_state_msgs[0])
        self.covariance_pub_1.publish(robot_covariance_msgs[0])
        self.state_pub_2.publish(robot_state_msgs[1])
        self.covariance_pub_2.publish(robot_covariance_msgs[1])
        self.state_pub_3.publish(robot_state_msgs[2])
        self.covariance_pub_3.publish(robot_covariance_msgs[2])

def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
