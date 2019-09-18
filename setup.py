from setuptools import setup

setup(name='gym_mupen64plus',
      version='0.0.4',
      install_requires=['gym==0.17.2',
                        'numpy==1.19.4',
                        'PyYAML==5.3.1',
                        'termcolor==1.1.0',
                        'mss==4.0.2', # 4.0.3 removes support for Python 2.7
                        'opencv-python==4.4.0.46'])
