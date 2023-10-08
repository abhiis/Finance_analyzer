import pandas as pd
import io


def read_file(file_bytes):
    df = pd.read_csv(io.StringIO(file_bytes.decode('utf-8')))
    return df
