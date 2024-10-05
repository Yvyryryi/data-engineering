import pandas as pd
from numpy._typing import NDArray
import numpy as np

def get_metadata() -> pd.DataFrame:
    return pd.read_csv("data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv")
    
def preprocess_data_sample(sample_ind: int, metadata) -> float:
    df: pd.DataFrame = pd.read_csv(f"data/lunar/training/data/S12_GradeA/{metadata['filename'].iloc[sample_ind]}.csv", parse_dates =['time_abs(%Y-%m-%dT%H:%M:%S.%f)'] ,index_col = ['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])
    df.columns = ['time', 'v']
    return df, metadata['time_rel(sec)'].iloc[sample_ind]

def tv_analysis(chunk_size: int, v: NDArray) -> None:
    out = np.abs(v)
    for i in range(chunk_size):
        out = np.abs(out[1:] - out[:-1])
    return out