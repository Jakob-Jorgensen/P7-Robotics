import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jakob/Desktop/P7-Robotics/vision_toolbox/install/vision_toolbox'
