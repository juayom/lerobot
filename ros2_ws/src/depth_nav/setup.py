from setuptools import find_packages, setup

package_name = 'depth_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lerobot',
    maintainer_email='lerobot@todo.todo',
    description='Depth based navigation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'depth_detector = depth_nav.depth_detector:main',
            'robot_fsm = depth_nav.robot_fsm:main',
        ],
    },
)
