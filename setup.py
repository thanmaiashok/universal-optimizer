from setuptools import setup, find_packages

setup(
    name="predycat-ai",
    version="1.0.0",
    description="PredycatAI Universal Optimizer - Production-grade AI optimization system",
    author="Thanmai",
    author_email="thanmai5ashok@gmail.com",
    url="https://github.com/thanmai29/universal-optimizer",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.30.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
        ],
        "gpu": [
            "bitsandbytes>=0.41.0",
            "onnxruntime-gpu>=1.15.0",
        ],
        "mobile": [
            "optimum>=1.7.0",
            "accelerate>=0.20.0",
        ],
        "full": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "bitsandbytes>=0.41.0",
            "onnx>=1.14.0",
            "onnxruntime-gpu>=1.15.0",
            "optimum>=1.7.0",
            "accelerate>=0.20.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "predycat=predycat_ai.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)