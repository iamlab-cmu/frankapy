import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from franka_interface_msgs.msg import SensorDataGroup
import quaternion

class CollisionBoxesPublisher(Node):

    def __init__(self, topic_name, world_frame='panda_link0'):
        super().__init__('collision_boxes_publisher')
        self._boxes_pub = self.create_publisher(MarkerArray, topic_name, 10)
        self._world_frame = world_frame

    def publish_boxes(self, boxes):
        markers = []
        for i, box in enumerate(boxes):
            marker = Marker()
            marker.type = Marker.CUBE
            marker.header.stamp = self.get_clock().now()
            marker.header.frame_id = self._world_frame
            marker.id = i

            marker.lifetime = rclpy.time.Duration()

            marker.pose.position.x = box[0]
            marker.pose.position.y = box[1]
            marker.pose.position.z = box[2]

            marker.scale.x = box[-3]
            marker.scale.y = box[-2]
            marker.scale.z = box[-1]

            if len(box) == 9:
                q = quaternion.from_euler_angles(box[3], box[4], box[5])
                for k in 'wxyz':
                    setattr(marker.pose.orientation, k, getattr(q, k))
            elif len(box) == 10:
                for j, k in enumerate('wxyz'):
                    setattr(marker.pose.orientation, k, box[3 + j])
            else:
                raise ValueError('Invalid format for box!')

            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.color.a = 0.6

            markers.append(marker)

        marker_array = MarkerArray(markers)
        self._boxes_pub.publish(marker_array)