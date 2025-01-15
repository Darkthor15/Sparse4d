import rclpy
import time
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from sparse_msgs.msg import CustomTFMessage

class TFMsg(Node):
    def __init__(self):
        super().__init__('tfmsg_node')

        self.subs = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10,
        )

        self.pubs = self.create_publisher(
            CustomTFMessage,
            '/tf_stamped',
            10,
        )
        self.message_time_threshold = 30
        self.message_time = time.time()
        self.dest_timer = self.create_timer(1.0, self.check_timeout)

    def tf_callback(self, msg):
        self.message_time = time.time()
        out = CustomTFMessage()
        out.tf_message = msg
        out.header = msg.transforms[0].header
        self.pubs.publish(out)
    
    def check_timeout(self):
        curr_time = time.time()
        time_since_last = curr_time - self.message_time
        if time_since_last > self.message_time_threshold:
            self.get_logger().info(f"No messages from topics for {self.message_time_threshold}[sec].")
            self.destroy_node()
        
    def destroy_node(self):
        self.get_logger().info("Exiting.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    tfmsg_node = TFMsg()
    rclpy.spin(tfmsg_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()