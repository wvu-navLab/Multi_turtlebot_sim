cmake_minimum_required(VERSION 3.8)
project(multi_turtlebot_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/RobotState.msg
  msg/RobotCovariance.msg
  DEPENDENCIES std_msgs
)
ament_export_dependencies(rosidl_default_runtime)

install(PROGRAMS
  scripts/waypoint_follower.py
  DESTINATION lib/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/Robot_state_publisher.py
  DESTINATION lib/${PROJECT_NAME}/
)

install(PROGRAMS
scripts/uwb_node.py
DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
scripts/imu_node.py
DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/ekf.py
  DESTINATION lib/${PROJECT_NAME}/
)

install(DIRECTORY launch meshes urdf worlds models param map
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
