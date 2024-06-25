#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np

class ImuNode(Node):

    def __init__(self):
        super().__init__('imu_node')
        self.declare_parameter('arw_var', 4.36e-7)
        self.declare_parameter('bias_var', 2.9e-5)
        self.declare_parameter('dt', 0.01)

        self.arw_var = self.get_parameter('arw_var').value
        self.bias_var = self.get_parameter('bias_var').value
        self.dt = self.get_parameter('dt').value

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/robot1/gazebo_ros_imu/out',
            self.imu_callback,
            10
        )

        self.imu_publisher = self.create_publisher(Imu, 'imu/processed', 10)
        self.bias_angular = np.zeros(3)
        self.bias_linear = np.zeros(3)

    def imu_callback(self, msg):
        processed_msg = Imu()
        processed_msg.header = msg.header
        processed_msg.orientation = msg.orientation

        processed_msg.angular_velocity.x = self.process_measurement(
            msg.angular_velocity.x, 0)
        processed_msg.angular_velocity.y = self.process_measurement(
            msg.angular_velocity.y, 1)
        processed_msg.angular_velocity.z = self.process_measurement(
            msg.angular_velocity.z, 2)

        processed_msg.linear_acceleration.x = self.process_measurement(
            msg.linear_acceleration.x, 0, is_angular=False)
        processed_msg.linear_acceleration.y = self.process_measurement(
            msg.linear_acceleration.y, 1, is_angular=False)
        processed_msg.linear_acceleration.z = self.process_measurement(
            msg.linear_acceleration.z, 2, is_angular=False)

        self.imu_publisher.publish(processed_msg)

    def process_measurement(self, value, index, is_angular=True):
        arw = np.random.normal(0, np.sqrt(self.arw_var))
        bias_instability = self.dt * np.random.normal(0, np.sqrt(self.bias_var))

        if is_angular:
            bias = self.bias_angular
        else:
            bias = self.bias_linear

        measurement = value + bias_instability + arw
        bias[index] += bias_instability

        return measurement

def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
