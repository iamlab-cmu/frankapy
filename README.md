# frankapy

## Requirements

* A computer with the Ubuntu 16.04 / 18.04.
* ROS Kinetic / Melodic
* [Protocol Buffers](https://github.com/protocolbuffers/protobuf)

## Install ProtoBuf

** This is only needed if you plan to modify the proto messages. You don't need to install or compile protobuf for using frankapy **

1. Read installation instructions here https://github.com/protocolbuffers/protobuf/blob/master/src/README.md.

2. We use both C++ and Python versions of protobufs so you would need to install Protobufs from source. In short you will have to do the following. NOTE: However, make to read protobuf installation instructions once.

Do `nproc` to find out how many cores you have, and use that as the `N` number in the `make` command below:

```shell
sudo apt-get install autoconf automake libtool curl make g++ unzip
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protobuf-all-3.11.4.zip
unzip protobuf-all-3.11.4.zip
cd protobuf-3.11.4
./configure
make -jN
make check -jN
sudo make install
sudo ldconfig
```

## Installation

1. Clone Repo and its Submodules:

   ```bash
   git clone --recurse-submodules https://github.com/iamlab-cmu/frankapy.git
   ```
All directories below are given relative to `/frankapy`.

2. First source into your virtualenv or conda env (should be Python 3.6). Then:
   ```bash
   pip install -e .
   ```
   
3. To compile the catkin_ws use the following script:
	```bash
   ./bash_scripts/make_catkin.sh
   ```


4. To make the protobufs use the following script:
	```bash
   ./bash_scripts/compile_proto.sh
   ```

## Setting Up SSH Key to Control PC
1. Generate an ssh key by executing the following commands or reading the instructions here: https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   [Press enter]
   [Press enter]
   [Press enter]
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_rsa
   ```
2. Upload your public ssh key to the control pc.
   1. In a new terminal, ssh to the control PC.
      ```bash
      ssh iam-lab@iam-[control-pc-name]
      Input password to control-pc.
      ```
   2. Use your favorite text editor to open the authorized_keys file.
      ```bash
      vim ~/.ssh/authorized_keys
      ```
   3. In a separate terminal on your Workhorse PC, use your favorite text editor to open your id_rsa.pub file.
      ```bash
      vim ~/.ssh/id_rsa.pub
      ```
   4. Copy the contents from your id_rsa.pub file to a new line on the authorized_keys file on the Control PC. Then save. 
   5. Open a new terminal and try sshing to the control PC and it should no longer require a password. 
3. (Optional) Upload your ssh key to github by following instructions here: https://help.github.com/en/articles/adding-a-new-ssh-key-to-your-github-account

## Unlocking the Franka Robot
1. In a new terminal, ssh to the control PC with option -X.
   ```bash
   ssh -X iam-lab@iam-[control-pc-name]
   ```
2. Open a web browser, either firefox or google chrome.
   ```bash
   firefox
   ```
3. Go to 172.16.0.2 in the web browser.
4. (Optional) Input the username admin and the password to login to the Franka Desk GUI.
5. Unlock the robot by clicking the unlock button on the bottom right of the web interface.
6. If the robot has pink lights, press down on the e-stop and then release it and the robot should turn blue. If the robot is white, just release the e-stop and it should also turn blue.

## Running the Franka Robot

1. Make sure that both the user stop and the brakes of the Franka robot have been unlocked in the Franka Desk GUI.
2. Open up a new terminal and go to the frankapy directory.
   ```bash
   bash ./bash_scripts/start_control_pc.sh -i iam-[control-pc-name]
   ```
3. Open up a new terminal and go to the frankapy directory. Be in the same virtualenv or Conda env that FrankaPy was installed in. Place your hand on top of the e-stop. Reset the robot pose with the following command.
   ```bash
   python scripts/reset_arm.py
   ```
   
See example scripts in the `examples/` and `scripts/` to learn how to use the `FrankaPy` python package.
