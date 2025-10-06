Só executar esses 2 comandos no terminal do Ubuntu 24.04 com ROS2 Jazzy (1 em cada terminal).

ros2 launch denso_robot_bringup denso_robot_bringup.launch.py sim_gazebo:=true robot_controller:=trajectory_controller

ros2 launch denso_robot_moveit_config denso_robot_moveit.launch.py use_sim_time:=true

