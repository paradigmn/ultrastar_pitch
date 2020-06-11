from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='ultrastar_pitch',
      version='0.73',
      description='An attempt to automate the pitch detection for USDX projects',
      long_description=readme(),
      classifiers=[
        'Intended Audience :: End Users/Desktop',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      url='https://github.com/paradigmn/ultrastar_pitch',
      author='paradigm',
      author_email='ultrastarpitch@gmail.com',
      packages=['ultrastar_pitch'],
      entry_points = {"console_scripts": ['ultrastar-pitch = ultrastar_pitch.ultrastar_pitch:main']},
      install_requires=['numpy',
                        'tensorflow>=2.0'],
      include_package_data=True,
      zip_safe=False)
