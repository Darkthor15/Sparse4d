import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from sparse_msgs.msg import BBoxes3D  # Adjust to your package's namespace

class PredictedBBoxVisualizer(Node):
    def __init__(self):
        super().__init__('predicted_bbox_visualizer')
        self.subscription = self.create_subscription(
            BBoxes3D,
            '/bboxes3d',
            self.listener_callback,
            10)

        self.publisher = self.create_publisher(MarkerArray, '/markers/predictions', 10)
        
        # Set a timer to publish at ~0.6 Hz (1.66 seconds per interval)
        self.timer = self.create_timer(1.66, self.publish_markers)
        self.marker_array = MarkerArray()

    def listener_callback(self, msg):
        # Clear marker array for each new message batch
        self.marker_array.markers.clear()

        for box in msg.boxes3d:
            marker = Marker()
            marker.header.frame_id = "base_link" #LIDAR_TOP  
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "pred_bbox3d"
            marker.id = box.id  # Assign unique ID from the box

            # Set marker type to CUBE
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position from bbox (center x, y, z)
            marker.pose.position.x = box.bbox[0]
            marker.pose.position.y = box.bbox[1]
            marker.pose.position.z = box.bbox[2]

            # Dimensions (width, height, depth)
            marker.scale.x = box.bbox[4]
            marker.scale.y = box.bbox[3]
            marker.scale.z = box.bbox[5]

            # Orientation quaternion (x, y, z, w)
            marker.pose.orientation.x = box.bbox[7]
            marker.pose.orientation.y = box.bbox[8]
            marker.pose.orientation.z = box.bbox[9]
            marker.pose.orientation.w = box.bbox[6]

            # Color for predicted bounding boxes (different from ground truth)
            marker.color.a = 0.5  # Semi-transparent
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0  # Blue color for predictions

            self.marker_array.markers.append(marker)

    def publish_markers(self):
        # Publish the marker array at the defined frequency (0.6 Hz)
        self.publisher.publish(self.marker_array)

def main(args=None):
    rclpy.init(args=args)
    predicted_bbox_visualizer = PredictedBBoxVisualizer()
    rclpy.spin(predicted_bbox_visualizer)
    predicted_bbox_visualizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

