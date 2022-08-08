import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from franka_interface_msgs.srv import GetCurrentGripperState


class GripperStateClient(Node):

    def __init__(self, gripper_state_server_name='/get_current_gripper_state_server_node_1/get_current_gripper_state_server', offline=False):
        super().__init__('gripper_state_client')
        
        print(gripper_state_server_name)

        self._offline = offline
        if not self._offline:
            self._gripper_state_client = self.create_client(GetCurrentGripperState, gripper_state_server_name)
            while not self._gripper_state_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Get Current Gripper State Service is not available, waiting again...')
            self.req = GetCurrentGripperState.Request()

    def get_current_gripper_state(self):
        if self._offline:
            current_gripper_state = JointState()
            return current_gripper_state

        self.future = self._gripper_state_client.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result().gripper_state