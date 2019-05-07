from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='ultrastar_pitch',
      version='0.21',
      description='An attempt to automate the pitch detection for USDX projects',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.7',
      ],
      url='https://github.com/paradigmn/ultrastar_pitch',
      author='paradigm',
      author_email='ultrastarpitch@gmail.com',
      license='GPL2.0',
      packages=['upitch'],
      install_requires=['numpy',
                        'scipy',],
      include_package_data=True,
      zip_safe=False)

