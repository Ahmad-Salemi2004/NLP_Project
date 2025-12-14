#!/usr/bin/env python3

from setuptools import setup, find_packages
import os
import sys

# Check Python version
if sys.version_info < (3, 8):
    print("❌ Error: Python 3.8 or higher is required")
    print(f"    You are using Python {sys.version_info.major}.{sys.version_info.minor}")
    sys.exit(1)

# Read the README file for long description
def read_file(filename):
    """Read content from a file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Text Summarization NLP Project using BART model"

# Get long description from README
long_description = read_file("README.md")

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt"""
    requirements = []
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    # Remove version comments
                    if " #" in line:
                        line = line.split(" #")[0].strip()
                    requirements.append(line)
    except FileNotFoundError:
        print("⚠️  requirements.txt not found, using default requirements")
        requirements = [
            "flask>=2.3.0",
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "datasets>=2.14.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "sentencepiece>=0.1.99",
            "accelerate>=0.24.0",
            "evaluate>=0.4.0",
            "rouge-score>=0.1.2",
            "tqdm>=4.65.0",
            "pyyaml>=6.0.0",
        ]
    
    return requirements

# Get project version
def get_version():
    """Get version from __init__.py or use default"""
    try:
        # Try to get version from src/__init__.py
        init_path = os.path.join("src", "__init__.py")
        with open(init_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    # Get value from __version__ = "1.0.0"
                    return line.split("=")[1].strip().strip('"\'')
    except FileNotFoundError:
        pass
    
    # Default version
    return "1.0.0"

# Package data (non-Python files to include)
def get_package_data():
    """Get package data files to include"""
    package_data = {
        "": [
            "*.md",           # README files
            "*.txt",          # Text files
            "*.yaml",         # Config files
            "*.yml",          # Config files
            "*.json",         # JSON files
        ]
    }
    
    # Check if directories exist and add their patterns
    if os.path.exists("configs"):
        package_data[""].append("configs/*.yaml")
        package_data[""].append("configs/*.yml")
    
    if os.path.exists("templates"):
        package_data[""].append("templates/*.html")
        package_data[""].append("templates/**/*.html")
    
    if os.path.exists("static"):
        package_data[""].append("static/**/*")
    
    return package_data

# Entry points (console scripts)
def get_entry_points():
    """Get console script entry points"""
    entry_points = {
        "console_scripts": [
            # Training
            "summarize-train = src.train:main",
            "train-summarizer = src.train:main",
            
            # Inference
            "summarize-text = src.inference:main",
            "summarize = src.inference:main",
            "summarize-cli = src.cli:main",
            
            # Evaluation
            "summarize-eval = src.evaluate:main",
            "evaluate-summarizer = src.evaluate:main",
            
            # Web app
            "summarize-web = app:main",
            "run-summarizer = app:main",
            
            # Utilities
            "summarize-download = src.utils:download_data",
            "summarize-config = src.utils:show_config",
        ]
    }
    return entry_points

# Project classifiers
classifiers = [
    # Development status
    "Development Status :: 4 - Beta",
    
    # Intended audience
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    
    # Topics
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Text Processing :: Linguistic",
    "Topic :: Education",
    
    # License
    "License :: OSI Approved :: MIT License",
    
    # Programming languages
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    
    # Operating systems
    "Operating System :: OS Independent",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS",
    
    # Frameworks
    "Framework :: Flask",
    
    # Natural Language
    "Natural Language :: English",
]

# Main setup configuration
setup(
    # Basic information
    name="text-summarization-nlp",
    version=get_version(),
    author="[Your Name]",
    author_email="[your.email@example.com]",
    
    # Description
    description="Text Summarization using Fine-tuned BART Model - NLP College Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Project URLs
    url="https://github.com/yourusername/nlp-text-summarization",
    project_urls={
        "Homepage": "https://github.com/yourusername/nlp-text-summarization",
        "Documentation": "https://github.com/yourusername/nlp-text-summarization#readme",
        "Bug Reports": "https://github.com/yourusername/nlp-text-summarization/issues",
        "Source Code": "https://github.com/yourusername/nlp-text-summarization",
    },
    
    # Package discovery
    packages=find_packages(
        include=["src", "src.*", "app"],
        exclude=["tests", "tests.*", "docs", "docs.*"]
    ),
    
    # Package data
    package_data=get_package_data(),
    include_package_data=True,
    
    # Requirements
    install_requires=read_requirements(),
    
    # Optional requirements (extras)
    extras_require={
        # Development extras
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        
        # Web extras
        "web": [
            "flask-cors>=4.0.0",
            "flask-swagger-ui>=4.11.0",
            "gunicorn>=21.0.0",
            "gevent>=23.0.0",
        ],
        
        # Data extras
        "data": [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
        
        # Model extras
        "model": [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "accelerate>=0.24.0",
            "sentencepiece>=0.1.99",
        ],
        
        # Full installation (everything)
        "full": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.4.0",
            "flask-cors>=4.0.0",
            "gunicorn>=21.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "accelerate>=0.24.0",
            "evaluate>=0.4.0",
            "rouge-score>=0.1.2",
        ],
    },
    
    # Entry points
    entry_points=get_entry_points(),
    
    # Classifiers
    classifiers=classifiers,
    
    # Python requirements
    python_requires=">=3.8",
    
    # License
    license="MIT",
    
    # Keywords
    keywords=[
        "nlp",
        "text-summarization",
        "bart",
        "transformers",
        "huggingface",
        "deep-learning",
        "machine-learning",
        "college-project",
        "education",
        "flask",
        "web-app",
    ],
    
    # Platforms
    platforms=["any"],
    
    # Scripts (if any)
    scripts=[
        # You can add script files here if you have any
        # "scripts/summarize.sh",
        # "scripts/train_model.py",
    ],
    
    # Zip safe
    zip_safe=False,
    
    # Options
    options={
        "bdist_wheel": {
            "universal": False,  # Pure Python, but version specific
        },
        "build": {
            "build_base": "build",
        },
    },
)

# Print success message
print("\n" + "="*60)
print("✅ setup.py configured successfully!")
print("="*60)
print("\nYou can now install the package using:")
print("  pip install -e .                    # Editable installation")
print("  pip install .                       # Regular installation")
print("  pip install -e .[dev]               # With dev dependencies")
print("  pip install -e .[full]              # With all dependencies")
print("\nAvailable commands after installation:")
print("  summarize-train                     # Train the model")
print("  summarize-text                      # Run inference")
print("  summarize-eval                      # Evaluate model")
print("  summarize-web                       # Run web app")
print("="*60 + "\n")
