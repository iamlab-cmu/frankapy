import rclpy
from rclpy.node import Node
from franka_interface_msgs.msg import FrankaInterfaceStatus
from franka_interface_msgs.srv import GetCurrentFrankaInterfaceStatus


class FrankaInterfaceStatusClient(Node):

    def __init__(self, franka_interface_status_server_name='/get_current_franka_interface_status_server', offline=False):
        super().__init__('franka_interface_status_client')

        self._offline = offline
        if not self._offline:
            self._franka_interface_status_client = self.create_client(GetCurrentFrankaInterfaceStatus, franka_interface_status_server_name)
            while not self._franka_interface_status_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Get Current Robot State Service is not available, waiting again...')
            self.req = GetCurrentFrankaInterfaceStatus.Request()

    def get_current_franka_interface_status(self):
        if self._offline:
            current_franka_interface_status = FrankaInterfaceStatus()
            current_franka_interface_status.is_ready = True
            current_franka_interface_status.is_fresh = True
            return current_franka_interface_status

        self.future = self._franka_interface_status_client.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result().franka_interface_status