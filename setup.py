from setuptools import setup, find_packages
# pip install "pydantic<2"
exec(open('denoising_diffusion_pytorch/version.py').read())

setup(
  name = 'denoising-diffusion-pytorch',
  packages = find_packages(),
  version = __version__,
  license='MIT',
  description = 'Denoising Diffusion Probabilistic Models - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/denoising-diffusion-pytorch',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'generative models'
  ],
  install_requires=[
    'tqdm',
    'accelerate==0.27.2',
    'einops==0.7.0',
    'ema-pytorch==0.4.2',
    'pillow==10.2.0',
    'pytorch-fid==0.3.0',
    'natsort==8.1.0',
    'h5py==3.10.0',
    'pickle5==0.0.11',
    'vtk==9.3.1',
    'pyvista==0.43.3',
    'pyvistaqt==0.9.0',
    'ray==2.10.0',
    'protobuf==3.20.3',
    # 'PyQt5==5.15.10',
    # 'seaborn==0.13.2'
    'imageio==2.34.1',
    'imageio-ffmpeg==0.4.9',
    'urllib3==2.0.5',
    'imgsim==0.1.2',
    'scikit-image==0.21.0',
    'pydantic<2',
    'trimesh',
    'rtree',
    'typed-argument-parser',
    'gitpython',
    'einops',
    'ffmpeg',
    'ffprobe',
    'pillow',
    'tqdm',
    'pandas',
    'wandb',
    'crcmod ',  # for fast gsutil rsync on large files
    'cryptography',
    'tensorboardX',
    'h5py'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
