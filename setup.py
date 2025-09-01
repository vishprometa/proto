#!/usr/bin/env python3
"""
Setup script for Proto ClickHouse AI Agent
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="proto-clickhouse-agent",
    version="1.0.3",
    author="Proto Team",
    author_email="team@proto.dev",
    description="ClickHouse AI Agent - Natural language interface for ClickHouse databases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/proto",
    packages=find_packages(),
    # Ensure top-level CLI module is included so entry point works
    py_modules=["main"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "typer[all]>=0.9.0",
        "rich>=13.0.0",
        "textual>=0.40.0",
        "clickhouse-connect>=0.6.0",
        "clickhouse-driver>=0.2.0",
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "plotly>=5.0.0",
        "matplotlib>=3.5.0",
        "httpx>=0.24.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "aiofiles>=23.0.0",
        "orjson>=3.8.0",
        "structlog>=23.0.0",
        "openai>=1.0.0",
        "anthropic>=0.25.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "proto=main:app",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)