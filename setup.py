from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="FinanceAnalyzer",
    version="0.2.0",
    author="Abhishek Srivastava",
    author_email="abhiis@eleven11.pro",
    description="A Python library for analyzing financial data from bank statements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FinanceAnalyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "dash==2.13.0",
        "pandas==2.1.1",
        "plotly==5.3.1",
        "numpy==1.26.0",
        "pdfplumber==0.5.28",
        "PyPDF2==1.26.0",
        "tabula-py==2.3.0",
        "scipy==1.11.3",
        "openpyxl==3.1.2"
    ],
    python_requires=">=3.6",
)
