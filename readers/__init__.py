# readers/__init__.py

from .csv_reader import read_file as read_csv
from .pdf_reader import read_file as read_pdf
from .excel_reader import read_file as read_excel


def read_any(file_bytes, file_type, password=None):
    if file_type == 'csv':
        return read_csv(file_bytes)
    elif file_type == 'pdf':
        return read_pdf(file_bytes, password)
    elif file_type == 'xlsx':
        return read_excel(file_bytes)
    else:
        raise ValueError("Unsupported file type")
