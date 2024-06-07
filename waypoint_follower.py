import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
import math
import yaml
import os

class WaypointFollower(Node):
    def __init__(self, robot_id, waypoints):
        super().__init__(f'waypoint_follower_{robot_id}')

        self.robot_id = robot_id
        self.waypoints = waypoints  # List of Point objects
        self.waypoint_index = 0
        self.current_position = Point()  # Initialize as a Point object
        self.current_orientation = 0
        self.command_vel_pub = self.create_publisher(Twist, f'robot{robot_id}/cmd_vel', 10)
        self.odometry_sub = self.create_subscription(
            Odometry, f'/robot{robot_id}/odom', self.odometry_callback, 10)
        self.timer = self.create_timer(0.5, self.run)
        
    def odometry_callback(self, msg):
        self.current_position.x = msg.pose.pose.position.x
        self.current_position.y = msg.pose.pose.position.y
        self.current_orientation = self.quaternion_to_euler(msg.pose.pose.orientation)

    def quaternion_to_euler(self, quaternion):
        x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
        sin_r_cosp = 2 * (w * x + y * z)
        cos_r_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sin_r_cosp, cos_r_cosp)

        sin_p = 2 * (w * y - z * x)
        pitch = math.asin(sin_p)

        sin_y_cosp = 2 * (w * z + x * y)
        cos_y_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(sin_y_cosp, cos_y_cosp)

        return yaw

    def run(self):
        if self.waypoint_index < len(self.waypoints):
            next_waypoint = self.waypoints[self.waypoint_index]  # Access Point object
        else:
            # Stop the robot if all waypoints are reached
            twist_msg = Twist()
            twist_msg.angular.z = 0.0
            twist_msg.linear.x = 0.0
            self.command_vel_pub.publish(twist_msg)
            return
        
        dx = next_waypoint.x - self.current_position.x
        dy = next_waypoint.y - self.current_position.y
        target_angle = math.atan2(dy, dx)  # Calculate target angle

        # Calculate angular velocity based on angle deficit
        angle_deficit = target_angle - self.current_orientation
        while angle_deficit > math.pi:
            angle_deficit -= 2 * math.pi
        while angle_deficit < -math.pi:
            angle_deficit += 2 * math.pi
        angular_velocity = 0.5 * angle_deficit  # Adjust gain as needed

        # Create and publish Twist message
        twist_msg = Twist()

        # Apply threshold and set angular/linear velocities
        if abs(angle_deficit) > 0.1:
            twist_msg.angular.z = angular_velocity
            twist_msg.linear.x = 0.0  # Set linear to zero while adjusting angle
        else:
            twist_msg.angular.z = 0.0
            twist_msg.linear.x = 0.3  # Set linear velocity after reaching threshold

        self.command_vel_pub.publish(twist_msg)

        # Check if waypoint is reached
        distance_to_waypoint = math.sqrt((self.current_position.x - next_waypoint.x)**2 +
                                         (self.current_position.y - next_waypoint.y)**2)
        if distance_to_waypoint < 0.1:
            print(f'Waypoint {self.waypoint_index} reached by robot {self.robot_id}')
            self.waypoint_index += 1

def load_waypoints_from_yaml(file_path):
    with open(file_path, 'r') as file:
        waypoints_data = yaml.safe_load(file)
    
    waypoints = {}
    for robot, points in waypoints_data.items():
        waypoints[robot] = [Point(x=point['x'], y=point['y']) for point in points]
    
    return waypoints

def main(args=None):
    rclpy.init(args=args)
    try:
        # Define waypoints for each robot (modify these as needed)
        #robot1_waypoints = [Point(x=1.0, y=1.0), Point(x=2.0, y=2.0), Point(x=1.0, y=3.0)]
        #robot2_waypoints = [Point(x=3.0, y=1.0), Point(x=4.0, y=2.0), Point(x=3.0, y=3.0)]
        #robot3_waypoints = [Point(x=5.0, y=1.0), Point(x=6.0, y=2.0), Point(x=5.0, y=3.0)]
        yaml_file_path = os.path.join(os.path.dirname(__file__), 'waypoint.yaml')
        waypoints = load_waypoints_from_yaml(yaml_file_path)



        # Create and run waypoint follower nodes for each robot
        robot1_follower = WaypointFollower(1, waypoints['robot1'])
        robot2_follower = WaypointFollower(2, waypoints['robot2'])
        robot3_follower = WaypointFollower(3, waypoints['robot3'])


        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(robot1_follower)
        executor.add_node(robot2_follower)
        executor.add_node(robot3_follower)

        executor.spin()
       
    except Exception as e:
        print(f'An error occurred: {e}')
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()

    
    
