import rospy
from std_msgs.msg import String
from franka_action_lib_msgs.msg import SensorData
from proto import sensor_msg_pb2

def talker():
    sensor_msg = sensor_msg_pb2.BoundingBox()
    sensor_msg.name = "hello"
    sensor_msg.id = 0
    sensor_msg.x = 1
    sensor_msg.y = 2
    sensor_msg.w = 3
    sensor_msg.h = 4

    sensor_data_bytes = sensor_msg.SerializeToString()
    
    print('data_in_bytes: type: {}, data: {}'.format(
       type(sensor_data_bytes), sensor_data_bytes))

    rospy.init_node('talker', anonymous=True)
    
    pub = rospy.Publisher('dummy_sensor', SensorData, queue_size=1000)

    f_data = SensorData()
    f_data.sensorDataInfo = "BoundingBox"
    f_data.type = 5
    f_data.size = len(sensor_data_bytes)
    f_data.sensorData = sensor_data_bytes

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        rospy.loginfo("will send message")
        pub.publish(f_data)
        rate.sleep()

    #f_data.size =len10

    # rospy.init_node('dummy_sensor_bytes_publisher', anonymous=True)
    # rate = rospy.Rate(10) # 10hz
    # while not rospy.is_shutdown():
       

    #     print('data_in_bytes',data_in_bytes)

    #     hello_str = "hello world %s" % rospy.get_time()
    #     rospy.loginfo(hello_str)
    #     pub.publish(f_data)
    #     rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass