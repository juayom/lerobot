from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sensevoice_stt'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lerobot',
    maintainer_email='lerobot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['tests'],
    entry_points={
        'console_scripts': [
            'stt_node = sensevoice_stt.stt_node:main',
            'dialog_controller = sensevoice_stt.dialog_controller:main', # 💡 여기에 메인 두뇌 진입점 추가!
        ],
    },
)
