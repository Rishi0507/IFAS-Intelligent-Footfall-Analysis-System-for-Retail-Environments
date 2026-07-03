from setuptools import setup, find_packages

setup(
    name="ipd-cctv-gender",
    version="9.3.0",
    description="Retail CCTV gender analytics pipeline (no age)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "opencv-python-headless", "opencv-contrib-python-headless",
        "ultralytics", "torch", "torchvision",
        "transformers", "pillow", "numpy", "pyyaml", "matplotlib",
    ],
    entry_points={"console_scripts": ["ipd-cctv=cctv_gender:main"]},
)
