from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='ultrastar_pitch',
      version='0.63',
      description='An attempt to automate the pitch detection for USDX projects',
      long_description=readme(),
      classifiers=[
        'Intended Audience :: End Users/Desktop',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3.7',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      url='https://github.com/paradigmn/ultrastar_pitch',
      author='paradigm',
      author_email='ultrastarpitch@gmail.com',
      packages=['ultrastar_pitch'],
      entry_points = {"console_scripts": ['ultrastar-pitch = ultrastar_pitch.ultrastar_pitch:main']},
      install_requires=['numpy',
                        'scipy',
                        'tensorflow>=2.0'],
      include_package_data=True,
      zip_safe=False)
