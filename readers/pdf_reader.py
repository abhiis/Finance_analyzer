# analyser/readers/pdf_reader.py

import PyPDF2
import tabula
import pandas as pd
import io


def read_file(file_bytes, password=None):
    # Create a BytesIO object and load the PDF
    pdf_reader = PyPDF2.PdfFileReader(io.BytesIO(file_bytes))

    # Decrypt the PDF if a password is provided
    if password:
        pdf_reader.decrypt(password)

    all_tables = pd.DataFrame()
    columns_vals = []

    for page_num in range(len(pdf_reader.pages)):
        tables = tabula.read_pdf(
            io.BytesIO(file_bytes),
            pages=page_num+1,
            multiple_tables=False,
            password=password
        )

        table = tables[0].reset_index()
        if page_num == 0:
            columns_vals = table.iloc[0].values.tolist()
            table = table.iloc[1:]
        table.columns = columns_vals
        all_tables = pd.concat([all_tables, table], ignore_index=True)

    return all_tables
