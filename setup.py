from setuptools import setup, find_packages

setup(
    name="tsjacket",
    version="0.1.0",
    author="Chinmay Aswale",
    description="Constrained LLM decoding — invalid outputs don't get corrected, they never get produced.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/poly-chinmay/tsjacket",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Ignore",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
