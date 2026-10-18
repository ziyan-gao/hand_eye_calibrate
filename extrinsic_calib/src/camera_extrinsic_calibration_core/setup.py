from setuptools import setup

package_name = 'camera_extrinsic_calibration_core'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Apple',
    maintainer_email='zhanghaolan7@gmail.com',
    description='Camera extrinsic calibration backend',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'calibration_node = camera_extrinsic_calibration_core.node:main',
        ],
    },
)
