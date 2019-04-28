from setuptools import setup

setup(name='ultrastar_pitch',
      version='0.1',
      description='An attempt to automate the pitch detection for USDX projects',
      url='https://github.com/paradigmn/ultrastar_pitch',
      author='paradigm',
      author_email='tmp@example.com',
      license='GPL2.0',
      packages=['upitch'],
      install_requires=['pydub',
                        'numpy',
                        'scipy',],
      zip_safe=False)

