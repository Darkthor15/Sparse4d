#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class ImageCompressorNode(Node):
    def __init__(self):
        super().__init__('image_compressor_node2')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            '/CAMERA_FRONT_RIGHT/image_8bit_color', 
            self.image_callback,
            10
        )
        self.pub = self.create_publisher(
            CompressedImage,
            '/CAMERA_FRONT_RIGHT/image_rect_compressed',
            10
        )

    def image_callback(self, img_msg):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8', [cv2.IMWRITE_JPEG_QUALITY, 100])

        success, encoded_img = cv2.imencode('.jpg', cv_img)
        if not success:
            self.get_logger().error("Failed to compress image to JPEG")
            return

        compressed_msg = CompressedImage()
        compressed_msg.header = img_msg.header  
        compressed_msg.format = "jpeg"
        compressed_msg.header.frame_id = 'base_link_fr'
        compressed_msg.data = np.array(encoded_img).tobytes() 

        self.pub.publish(compressed_msg)
def main(args=None):
    rclpy.init(args=args)
    node = ImageCompressorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



