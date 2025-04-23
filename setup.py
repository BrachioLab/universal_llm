from setuptools import setup, find_packages

setup(
    name="unillm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "openai",
        "anthropic",
        "boto3",
        "python-dotenv",
        "google-generativeai",
    ],
    author="Adam Stein",
    author_email="steinad@seas.upenn.edu",
    description="A unified interface for various Large Language Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BrachioLab/universal_llm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)