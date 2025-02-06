from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sparse4dv3',
            namespace='tfmsg',
            executable='tfmsg_node',
            name='sparse4dv3'
        ),
        Node(
            package='sparse4dv3',
            namespace='sparse',
            executable='sparse_node',
            name='sparse4dv3'
        ),
    ])