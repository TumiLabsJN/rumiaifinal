"""
Setup configuration for Foundation package.
"""

from setuptools import setup, find_packages

setup(
    name="foundation",
    version="1.0.0",
    description="Foundation package for RumiAI ML Pipeline",
    author="RumiAI Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0.0",
    ],
)
