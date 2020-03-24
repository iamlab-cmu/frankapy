import rospy
from visualization_msgs.msg import Marker, MarkerArray
import quaternion


class BoxesPublisher:

    def __init__(self, name, world_frame='panda_link0'):
        self._boxes_pub = rospy.Publisher(name, MarkerArray, queue_size=10)
        self._world_frame = world_frame

    def publish_boxes(self, boxes):
        markers = []
        for i, box in enumerate(boxes):
            marker = Marker()
            marker.type = Marker.CUBE
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = self._world_frame
            marker.id = i

            marker.lifetime = rospy.Duration()

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