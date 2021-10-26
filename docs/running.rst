Running the Robot
=================

Unlocking the Franka Robot
--------------------------

1. If you are running franka-interface and frankapy on the same computer, you can skip to step 2. If you have a FrankaPy PC and a Control PC, first ssh to the Control PC from the FrankaPy PC using SSH with option ``-X``::

    ssh -X [control-pc-username]@[control-pc-name]

2. Open a web browser, either firefox or google chrome using the command line::
    
    firefox

3. Go to ``172.16.0.2`` in the web browser.

4. Login to the Franka Desk GUI using the username and password that you used during the initial robot setup.

5. Unlock the robot by clicking the unlock button on the bottom right of the web interface.

6. If the robot has pink lights, press down on the e-stop and then release it and the robot should turn blue. If the robot is white, just release the e-stop and it should also turn blue.


Starting the FrankaPy Interface
-------------------------------

1. Make sure that the Franka Robot has been unlocked in the Franka Desk GUI and has blue lights. 

2. Open up a new terminal and go to the frankapy directory.

3. If you are running franka-interface and frankapy on the same computer, run the following command::

    bash ./bash_scripts/start_control_pc.sh -i localhost

4. Otherwise run the following command::

    bash ./bash_scripts/start_control_pc.sh -u [control-pc-username] -i [control-pc-name]

5. Please see the ``bash_scripts/start_control_pc.sh`` bash script for additional arguments, including specifying a custom directory for where franka-interface is installed on the Control PC. 

6. Open up a new terminal, enter into the same virtual environment that FrankaPy was installed in, go to the frankapy directory, then::

    source catkin_ws/devel/setup.bash

7. Place your hand on top of the e-stop and reset the robot with the following command::

    python scripts/reset_arm.py

8. See example scripts in the ``examples/`` and ``scripts/`` folders to learn how to use the FrankaPy python package.

9. Please note that if you are using a custom gripper or no gripper, please set the ``with_gripper=True`` flag in ``frankapy/franka_arm.py`` to ``False`` as well as set the ``with_gripper=1`` flag in ``bash_scripts/start_control_pc.sh`` to ``0``.