import argparse
from frankapy import FrankaArm
import math
import rospy
from std_msgs.msg import String

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=10)
    args = parser.parse_args()

    import time

    print('Starting robot')
    HOME_JOINTS = [0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4]
    fa = FrankaArm()
    start_time = time.time()
    fa.run_dynamic_joint_position_interpolation(HOME_JOINTS,
                                                duration=args.time)
    print('Applying 0 force torque control for {}s'.format(args.time))
    end_time = time.time()
    print('Did continue after sending skill {:.6f}'.format(end_time-start_time))
    rospy.sleep(0.1)
    new_time = time.time()
    print("will: {:.6f}".format(new_time - end_time))
