Installation
============

Requirements
------------

* A computer with Ubuntu 18.04 / 20.04
* ROS Melodic / Noetic


Steps
-----

1. Clone the Frankapy Repository and its Submodules::

    git clone --recurse-submodules git@github.com:iamlab-cmu/frankapy.git

2. To allow asynchronous gripper commands, we use the ``franka_ros`` package, so install libfranka and franka_ros using the following command (Change melodic to noetic if you are on Ubuntu 20.04)::

    sudo apt install ros-melodic-libfranka ros-melodic-franka-ros

3. Create and enter a virtual environment (:ref:`Virtual Environment`) and then run the following commands::

    cd frankapy
    pip install -e .

4. Compile the catkin_ws using the following script::

    ./bash_scripts/make_catkin.sh

5. Afterwards source the ``catkin_ws`` using the following command::

    source catkin_ws/devel/setup.bash

6. It is a good idea to add the following line to the end of your ``~/.bashrc`` file::

    source /path/to/frankapy/catkin_ws/devel/setup.bash --extend


(Optional) Additional Steps
---------------------------

Protobuf
~~~~~~~~

If you plan on modifying the library and the protobuf messages, you will need to compile the `Google Protocol Buffer <https://developers.google.com/protocol-buffers>`_ library from scratch using the following instructions.

1. First determine the number of cores on your computer using the command::

    nproc

2. Execute the following commands::

    sudo apt-get install autoconf automake libtool curl make g++ unzip
    wget https://github.com/protocolbuffers/protobuf/releases/download/v21.8/protobuf-all-21.8.zip
    unzip protobuf-all-21.8.zip
    cd protobuf-21.8
    ./configure

3. Use the number that was previously printed out using the ``nproc`` command above and substitute it as ``N`` below::

    make -jN
    sudo make install
    sudo ldconfig

4. Afterwards, you can make the protobuf messages using the following script::

    ./bash_scripts/make_proto.sh


Virtual Environment
~~~~~~~~~~~~~~~~~~~

Note that these instructions work on Ubuntu 18.04. They might be slightly different for Ubuntu 20.04.

1. Install Python3.6::

    sudo apt install -y python3-distutils

2. Install Pip::

    curl https://bootstrap.pypa.io/get-pip.py | sudo -H python3.6

3. Install Virtual Environment and Other Useful Python Packages::

    sudo -H pip3.6 install numpy matplotlib virtualenv

4. Create a Virtual Environment for Frankapy::

    virtualenv -p python3.6 franka

5. Enter into the Virtual Environment::

    source franka/bin/activate

6. How to exit the Virtual Environment::

    deactivate
