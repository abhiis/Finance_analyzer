
import pandas as pd
import io


def read_file(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes))
    return df
