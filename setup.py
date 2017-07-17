from setuptools import setup

setup(
    name="smartmirror_face",
    version=0.1,
    description="Smartmirror Openface",
    author='Alexander Neumann',
    author_email='alneuman@techfak.uni-bielefeld.de',
    packages=["smartmirror_face"],
    setup_requires=['numpy', 'scipy'],
    install_requires=['pandas', 'scikit-learn', 'scipy', 'numpy'],
    entry_points={
      'console_scripts': [
          'smartface = smartmirror_face.main:start'
      ]
    },
    package_data={
        'smartmirror_face': [
            'lua/batch-represent.lua',
            'lua/dataset.lua',
            'lua/main.lua',
            'lua/opts.lua',
            'models/haarcascade_frontalface_default.xml',
            'models/nn4.small2.v1.t7',
            'models/shape_predictor_68_face_landmarks.dat',
        ],
    }
)
