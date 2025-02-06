from setuptools import setup
import os
from glob import glob

package_name = 'sparse4dv3'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='shubhendu.ranadive@fixstars.com',
    description='parse4Dv3 Inference for ROS2 package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sparse_node = sparse4dv3.sparse_node:main',
            'tfmsg_node = sparse4dv3.tfmsg_node:main',
        ],
    },
)
