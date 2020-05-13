# frankapy

## Requirements

* A computer with the Ubuntu 16.04 / 18.04.
* ROS Kinetic / Melodic
* [Protocol Buffers](https://github.com/protocolbuffers/protobuf)

## Computer Setup Instructions

This library is intended to be installed on any computer in the same ROS network with the computer that interfaces with the Franka (we call the latter the Control PC).
`FrankaPy` will send commands to [franka-interface](https://github.com/iamlab-cmu/franka-interface), which actually controls the robot.

## Install ProtoBuf

**This is only needed if you plan to modify the proto messages. You don't need to install or compile protobuf for using frankapy**

We use both C++ and Python versions of protobufs so you would need to install Protobufs from source. 

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
See detailed instructions [here](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md)

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

4. To make the protobufs use the following script (**you don't need to do this if you haven't modified the proto messages**):
	```bash
   ./bash_scripts/make_proto.sh
   ```

## Configuring the network with the Control PC
### Ethernet
1. If you have an ethernet cable directly between the Control PC and the one sending Frankapy commands, you can go into the Ubuntu Network Connections Menu on the Control PC.
2. Select the Ethernet connection that corresponds to the port that you plugged the ethernet cable into and then click edit.
3. Go to the IPv4 Settings Tab and switch from Automatic (DHCP) to Manual.
4. Add a static ip address like 192.168.1.3 on the Control PC with netmask 24 and then click save.
5. Then do the same on the FrankaPy PC but instead set the static ip address to be 192.168.1.2.

### Wifi
1. If you are only communicating with the Control PC over Wifi, use the command `ifconfig` in order to get the wifi ip address of both computers and note them down.

### Editing the /etc/hosts file
1. Now that you have the ip addresses for both the Control PC and FrankaPy PC, you will need to edit the /etc/hosts files on both in order to allow communication between the 2 over ROS.
2. On the Control PC, run the command: `sudo gedit /etc/hosts`
3. If you are using an Ethernet connection, then add the following above the line `# The following lines are desirable for IPv6 capable hosts`:
   ```bash
   192.168.1.2     [frankapy-pc-name]
   ```
   Otherwise substitute 192.168.1.2 with the ip address of the FrankaPy PC that you discovered using the command `ifconfig`.
4. Afterwards, on the FrankaPy PC, again run the command `sudo gedit /etc/hosts` and add the line:
   ```bash
   192.168.1.3     [control-pc-name]
   ```
   Otherwise substitute 192.168.1.3 with the ip address of the Control PC that you discovered using the command `ifconfig`.
5. Now you should be able to ssh between the FrankaPy PC and the Control PC using the command:
   ```bash
   ssh [control-pc-username]@[control-pc-name]
   Input password to control-pc.
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
      ssh [control-pc-username]@[control-pc-name]
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
   ssh -X [control-pc-username]@[control-pc-name]
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
   bash ./bash_scripts/start_control_pc.sh -i [control-pc-name]
   ```
   Please see the `start_control_pc.sh` bash script for additional arguments, including specifying a custom directory for where `franka-interface` is installed on the Control PC as well as the username of the account on the Control PC. By default the username is `iam-lab`.
   
3. Open up a new terminal and go to the frankapy directory. Be in the same virtualenv or Conda env that FrankaPy was installed in. Place your hand on top of the e-stop. Reset the robot pose with the following command.
   ```bash
   python scripts/reset_arm.py
   ```
   
See example scripts in the `examples/` and `scripts/` to learn how to use the `FrankaPy` python package.
