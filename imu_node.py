#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np

class ImuNode(Node):

    def __init__(self):
        super().__init__('imu_node')
        
        # Declare and get parameters http://www.analog.com/media/en/technical-documentation/data-sheets/ADIS16488.pdf
        # Based on https://github.com/Alvinlyx/NaveGo/blob/master/%23navego_example.m%23; 
        self.declare_parameter('arw', [0.3, 0.3, 0.3])  # Angular Random Walk (deg/root-hour)
        self.declare_parameter('vrw', [0.029, 0.029, 0.029])  # Velocity Random Walk (m/s/root-hour)
        self.declare_parameter('gb_fix', [0.2, 0.2, 0.2])  # Gyro Static Bias (deg/s)
        self.declare_parameter('ab_fix', [16, 16, 16])  # Accel Static Bias (mg)
        self.declare_parameter('gb_drift', [6.25/3600, 6.25/3600, 6.25/3600])  # Gyro Dynamic Bias (deg/s)
        self.declare_parameter('ab_drift', [0.1, 0.1, 0.1])  # Accel Dynamic Bias (mg)
        self.declare_parameter('gb_corr', [100, 100, 100])  # Gyro Correlation Time (seconds)
        self.declare_parameter('ab_corr', [100, 100, 100])  # Accel Correlation Time (seconds)
        self.declare_parameter('dt', 0.01)  # Time step (seconds)

        # Convert parameters to required units
        self.arw = np.radians(np.array(self.get_parameter('arw').value) / 60)  # ARW: deg/root-hour -> rad/s/root-Hz
        self.vrw = np.array(self.get_parameter('vrw').value) / 60  # VRW: m/s/root-hour -> m/s²/root-Hz
        self.gb_fix = np.radians(np.array(self.get_parameter('gb_fix').value))  # Gyro Static Bias: deg/s -> rad/s
        self.ab_fix = np.array(self.get_parameter('ab_fix').value) * 0.001 * 9.81  # Accel Static Bias: mg -> m/s²
        self.gb_drift = np.radians(np.array(self.get_parameter('gb_drift').value))  # Gyro Dynamic Bias: deg/s -> rad/s
        self.ab_drift = np.array(self.get_parameter('ab_drift').value) * 0.001 * 9.81  # Accel Dynamic Bias: mg -> m/s²
        self.gb_corr = np.array(self.get_parameter('gb_corr').value)  # Gyro Correlation Time (seconds)
        self.ab_corr = np.array(self.get_parameter('ab_corr').value)  # Accel Correlation Time (seconds)
        self.dt = self.get_parameter('dt').value  # Time step (seconds)

        # IMU data subscription and publication
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/robot1/gazebo_ros_imu/out',
            self.imu_callback,
            10
        )
        self.imu_publisher = self.create_publisher(Imu, 'imu/processed', 10)

        # Initialize dynamic bias arrays for gyroscope and accelerometer
        self.gyro_dynamic_bias = np.zeros(3)
        self.accel_dynamic_bias = np.zeros(3)

    def imu_callback(self, msg):
        # Create a new IMU message and set the header and orientation
        processed_msg = Imu()
        processed_msg.header = msg.header
        processed_msg.orientation = msg.orientation

        # Process and add noise and biases to angular velocity
        processed_msg.angular_velocity.x = self.process_angular_velocity(
            msg.angular_velocity.x, 0)
        processed_msg.angular_velocity.y = self.process_angular_velocity(
            msg.angular_velocity.y, 1)
        processed_msg.angular_velocity.z = self.process_angular_velocity(
            msg.angular_velocity.z, 2)

        # Process and add noise and biases to linear acceleration
        processed_msg.linear_acceleration.x = self.process_linear_acceleration(
            msg.linear_acceleration.x, 0)
        processed_msg.linear_acceleration.y = self.process_linear_acceleration(
            msg.linear_acceleration.y, 1)
        processed_msg.linear_acceleration.z = self.process_linear_acceleration(
            msg.linear_acceleration.z, 2)

        # Publish the processed IMU message
        self.imu_publisher.publish(processed_msg)

    def process_angular_velocity(self, value, index):
        # Add Angular Random Walk (ARW) noise
        arw = np.random.normal(0, self.arw[index] / np.sqrt(self.dt))
        # Add dynamic bias (modeled with Gauss-Markov process)
        bias_instability = self.update_dynamic_bias(self.gyro_dynamic_bias, index, self.gb_drift, self.gb_corr)
        # Add static bias
        static_bias = self.gb_fix[index]
        return value + arw + bias_instability + static_bias

    def process_linear_acceleration(self, value, index):
        # Add Velocity Random Walk (VRW) noise
        vrw = np.random.normal(0, self.vrw[index] / np.sqrt(self.dt))
        # Add dynamic bias (modeled with Gauss-Markov process)
        bias_instability = self.update_dynamic_bias(self.accel_dynamic_bias, index, self.ab_drift, self.ab_corr)
        # Add static bias
        static_bias = self.ab_fix[index]
        return value + vrw + bias_instability + static_bias

    def update_dynamic_bias(self, bias_array, index, drift, corr_time):
        # Gauss-Markov process for simulating dynamic bias
        beta = self.dt / corr_time[index]
        sigma = drift[index]
        a1 = np.exp(-beta)
        a2 = sigma * np.sqrt(1 - np.exp(-2 * beta))
        bias_array[index] = a1 * bias_array[index] + a2 * np.random.normal()
        return bias_array[index]

def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
