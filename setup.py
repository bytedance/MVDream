from setuptools import setup, find_packages

setup(
    name='mvdream',
    version='0.0.1',
    description='Multi-view Diffusion Models',
    author="ByteDance",
    packages=find_packages(),
    package_data={"mvdream": ["configs/*.yaml"]} ,
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'omegaconf',
        'einops',
        'huggingface_hub',
        "transformers",
        "open-clip-torch",
    ],
)
