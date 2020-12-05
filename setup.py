from setuptools import setup
import os

def readme():
    with open('README.rst') as f:
        return f.read()
    
version = {}
with open(os.path.join(os.path.dirname(__file__), "ultrastar_pitch", "version.py")) as fp:
    exec(fp.read(), version)

setup(name='ultrastar_pitch',
      version=version["__version__"],
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
                        'onnxruntime'],
      include_package_data=True,
      zip_safe=False)
