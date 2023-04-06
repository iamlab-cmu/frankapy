# moveit_frankapy
1. Build the Docker Container:

    ```
    docker build -t moveit_frankapy .
    ```

2. Run Docker Container:
    ```
    cd moveit_frankapy
    bash run_docker.sh
    ```

    Run A Terminal connected to the Docker Container:
    ```
    cd moveit_frankapy
    bash terminal_docker.sh
    ```

3. Run the MoveIt launch file <br>
    In a terminal connected to the Docker Container:
    ```
    roslaunch manipulation demo_frankapy.launch
    ```

4. Run the demo_moveit.py script <br>
    In a terminal connected to the Docker Container:
    ```
    rosrun manipulation demo_moveit.py
    ```

