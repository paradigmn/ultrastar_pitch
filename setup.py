from setuptools import setup

setup(name='ultrastar_pitch',
      version='0.2',
      description='An attempt to automate the pitch detection for USDX projects',
      url='https://github.com/paradigmn/ultrastar_pitch',
      author='paradigm',
      author_email='ultrastarpitch@gmail.com',
      license='GPL2.0',
      packages=['upitch'],
      install_requires=['numpy',
                        'scipy',],
      zip_safe=False)

