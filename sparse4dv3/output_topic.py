#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# Message types
from sparse_msgs.msg import BBoxes3D, BoxInfo

class ThresholdedBBoxesNode(Node):
    def __init__(self):
        super().__init__('thresholded_bboxes_node')
        
        # Declare (or set) your threshold here
        self.score_threshold = 0.35

        # Subscriber: listens to the /bboxes3d topic
        self.sub = self.create_subscription(
            BBoxes3D,
            '/bboxes3d',
            self.bboxes_callback,
            10
        )

        # Publisher: publishes filtered bboxes to /thresholdedbbx
        self.pub = self.create_publisher(
            BBoxes3D,
            '/thresholdedbbx',
            10
        )

        self.get_logger().info(
            f'Node initialized. Listening on /bboxes3d and publishing thresholded results on /thresholdedbbx.\n'
            f'Confidence threshold = {self.score_threshold}'
        )

    def bboxes_callback(self, msg: BBoxes3D):
        """Filter out boxes below self.score_threshold, publish the rest."""
        filtered_bboxes = BBoxes3D()
        
        # Copy header from the original message
        filtered_bboxes.header = msg.header

        # Iterate through all boxes in the incoming message
        for box_info in msg.boxes3d:
            if box_info.score >= self.score_threshold:
                filtered_bboxes.boxes3d.append(box_info)

        # Log how many are published
        self.get_logger().info(
            f'Received {len(msg.boxes3d)} boxes, publishing {len(filtered_bboxes.boxes3d)} '
            'above threshold.'
        )

        # Publish
        self.pub.publish(filtered_bboxes)


def main(args=None):
    rclpy.init(args=args)
    node = ThresholdedBBoxesNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

