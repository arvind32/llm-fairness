# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open("/local/zemel/arvind/code/llm_fairness/README.md") as f:
    readme = f.read()

with open("/local/zemel/arvind/code/llm_fairness/LICENSE") as f:
    license = f.read()

setup(
    name="llm_fairness",
    version="0.0.1",
    description="llm_fairness",
    long_description=readme,
    author="anonymous",
    author_email="anonymous",
    url="anonymous",
    license=license,
    packages=find_packages(),
)