from setuptools import setup

package_name = 'vision_toolbox'

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
    maintainer='jakob',
    maintainer_email='jjorge21@student.aau.dk',
    description='Creating a toolbox that containes difrent ros2 tools',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 
            'Bag2png.Converter = ' + package_name+'.converter:main'
        ],
    },
)
