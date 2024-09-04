#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import numpy as np
from scipy.spatial.transform import Rotation as R
from multi_turtlebot_sim.msg import RobotState, RobotCovariance
from message_filters import Subscriber, ApproximateTimeSynchronizer


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
    gg = 9.8
    
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
    '''F21 = -skew_symmetric(np.dot(CbnPlus, accel))
    
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
    
    Q11 = np.random.normal(0,0.001)

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
    ])'''

        # Generate random Gaussian values for the diagonal elements
    Q_diag = np.zeros(15)
    Q_diag[0:3] = np.random.normal(0, Srg, 3)  # Attitude noise (due to gyro noise)
    Q_diag[3:6] = np.random.normal(0, Sra, 3)  # Velocity noise (due to acceleration noise)
    Q_diag[6:9] = np.random.normal(0, 0.0001, 3)  # Position noise (due to integrated velocity)
    Q_diag[9:12] = np.random.normal(0, Sbad, 3)  # Accel bias noise
    Q_diag[12:15] = np.random.normal(0, Sbgd, 3)  # Gyro bias noise

    # Build the diagonal Q matrix
    Q = np.diag(Q_diag)

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
        self.covariance = np.eye(45)

        # Define the variances for each state component
        attitude_variance = 0.1
        position_variance = 0.2
        velocity_variance = 0.01
        bias_variance = 0.0001

    # Loop through each robot and update the corresponding covariance block
        for i in range(3):
            start_idx = i * 15  
            self.covariance[start_idx:start_idx+3, start_idx:start_idx+3] = np.eye(3) * attitude_variance
            self.covariance[start_idx+3:start_idx+6, start_idx+3:start_idx+6] = np.eye(3) * position_variance
            self.covariance[start_idx+6:start_idx+9, start_idx+6:start_idx+9] = np.eye(3) * velocity_variance
            self.covariance[start_idx+9:start_idx+15, start_idx+9:start_idx+15] = np.eye(6) * bias_variance
        self.initialized = [False, False, False]
        self.f_ib_b = [np.zeros(3), np.zeros(3), np.zeros(3)]
        self.gyro_world = [np.zeros(3), np.zeros(3), np.zeros(3)]

    def create_subscriptions(self):
        self.imu_sub_1 = Subscriber(self, Imu, '/robot1/imu')
        self.imu_sub_2 = Subscriber(self, Imu, '/robot2/imu')
        self.imu_sub_3 = Subscriber(self, Imu, '/robot3/imu')
        self.sync = ApproximateTimeSynchronizer(
            [self.imu_sub_1, self.imu_sub_2, self.imu_sub_3],
            queue_size=10,
            slop=0.1  # 100ms tolerance for synchronization
        )
        self.sync.registerCallback(self.imu_callback)
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
    
    def imu_callback(self, imu_msg_1, imu_msg_2, imu_msg_3):
        # Process synchronized IMU messages from all three robots
        self.process_imu(imu_msg_1, 0)
        self.process_imu(imu_msg_2, 1)
        self.process_imu(imu_msg_3, 2)

    def predict(self, dt):
        g = np.array([0, 0, -9.8])  # gravity vector

        # Process the IMU data for each robot simultaneously
        roll1, pitch1, yaw1 = self.nominal_state[0], self.nominal_state[1], self.nominal_state[2]
        roll2, pitch2, yaw2 = self.nominal_state[15], self.nominal_state[16], self.nominal_state[17]
        roll3, pitch3, yaw3 = self.nominal_state[30], self.nominal_state[31], self.nominal_state[32]

        Cbn1 = R.from_euler('xyz', [roll1, pitch1, yaw1]).as_matrix()
        Cbn2 = R.from_euler('xyz', [roll2, pitch2, yaw2]).as_matrix()
        Cbn3 = R.from_euler('xyz', [roll3, pitch3, yaw3]).as_matrix()

        # Get the gyro and accel data for each robot
        gyro1 = self.gyro_world[0] - self.nominal_state[12:15]
        gyro2 = self.gyro_world[1] - self.nominal_state[27:30]
        gyro3 = self.gyro_world[2] - self.nominal_state[42:45]

        #print(self.f_ib_b)

        accel1 = self.f_ib_b[0] - self.nominal_state[9:12]
        accel2 = self.f_ib_b[1] - self.nominal_state[24:27]
        accel3 = self.f_ib_b[2] - self.nominal_state[39:42]

        # Update CbnPlus for each robot
        CbnPlus1 = Cbn1 @ (np.eye(3) + skew_symmetric(gyro1) * dt)
        CbnPlus2 = Cbn2 @ (np.eye(3) + skew_symmetric(gyro2) * dt)
        CbnPlus3 = Cbn3 @ (np.eye(3) + skew_symmetric(gyro3) * dt)

        # Velocity and Position Prediction for each robot
        for i, (CbnPlus, accel, idx) in enumerate(zip([CbnPlus1, CbnPlus2, CbnPlus3],
                                                    [accel1, accel2, accel3],
                                                    [0, 15, 30])):

            # Velocity prediction
            v_minus = self.nominal_state[idx+3:idx+6]
            v_plus = v_minus + ((CbnPlus @ accel) + g) * dt
            self.nominal_state[idx+3:idx+6] = v_plus

            # Position prediction
            r_minus = self.nominal_state[idx+6:idx+9]
            r_plus = r_minus + (dt / 2) * (v_minus + v_plus)
            self.nominal_state[idx+6:idx+9] = r_plus

        # Construct F matrices for each robot
        F1_block = construct_F(CbnPlus1, accel1)
        F2_block = construct_F(CbnPlus2, accel2)
        F3_block = construct_F(CbnPlus3, accel3)

        # Calculate Q matrices for each robot
        Q1 = calc_Q(dt, CbnPlus1, accel1)
        Q2 = calc_Q(dt, CbnPlus2, accel2)
        Q3 = calc_Q(dt, CbnPlus3, accel3)

        # Extend Q to 45x45 block diagonal matrix for 3 robots
        Q = np.block([
            [Q1, np.zeros((15, 15)), np.zeros((15, 15))],
            [np.zeros((15, 15)), Q2, np.zeros((15, 15))],
            [np.zeros((15, 15)), np.zeros((15, 15)), Q3]
        ])

        # Build the full F1 matrix
        F1_block = np.eye(15) + F1_block * dt
        F2_block = np.eye(15) + F2_block * dt
        F3_block = np.eye(15) + F3_block * dt

        F1 = np.block([
            [F1_block, np.zeros((15, 15)), np.zeros((15, 15))],
            [np.zeros((15, 15)), F2_block, np.zeros((15, 15))],
            [np.zeros((15, 15)), np.zeros((15, 15)), F3_block]
        ])

        # Update error state and covariance
        self.error_state = F1 @ self.error_state

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
            y = z - H @ self.error_state

        elif sensor_type == 'odom':
            idx = robot_idx1 * 15
            H = np.zeros((3, 45))  
            H[0, idx+3] = -1  # vx
            H[1, idx+4] = -1  # vy
            H[2, idx+5] = -1  # vz
            # Measurement residual for velocity
            y = z - H @ self.error_state # Only compare velocity

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

        #print(self.covariance)
        
        # Ensure S is positive definite before inverting
        #if np.all(np.linalg.eigvals(S) > 0):
        K = self.covariance @ H.T @ np.linalg.inv(S)
        #print(K)
        # Update the error state and covariance matrix
        self.error_state += K @ y
        #print(delta_error_state)
        I = np.eye(45)
        I_KH = I - K @ H
        self.covariance = (I_KH @ self.covariance @ np.transpose(I_KH)) + K @ R @ np.transpose(K)
        #else:
           #self.get_logger().warn('Measurement covariance matrix is not positive definite. Skipping update.')
          # return

        # Correct the nominal state with the error state
        self.nominal_state += self.error_state
        # Reset the error state
        self.error_state = np.zeros(45)

    def process_imu(self, msg, robot_index):
        imu_accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        
        imu_gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]) 
        imu_accel[2] = 9.8 + np.random.normal(0, 0.01)
        #print(imu_accel)

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
        R_mat = np.diag([0.01] * 3)  # Only 3x3 matrix for velocity noise
        self.update(odom_velocity, R_mat, 'odom', robot_index)


    def timer_callback(self):
        dt = 0.02  # Time step
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
