"""
FrankaPy Franka Panda Robot Control Wrapper
"""
from setuptools import setup

requirements = [
    'autolab_core',
    'numpy-quaternion',
    'numba',
    'rospkg',
    'catkin-tools'
]

setup(name='frankapy',
      version='0.0.0',
      description='FrankaPy Franka Panda Robot Control Wrapper',
      author='Mohit Sharma, Kevin Zhang, Jacky Liang',
      author_email='',
      package_dir = {'': '.'},
      packages=['frankapy'],
      install_requires = requirements,
      extras_require = {}
     )
