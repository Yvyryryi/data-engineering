import pandas as pd

def get_metadata() -> pd.DataFrame:
    return pd.read_csv("data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv")
    
def preprocess_data_sample(sample_ind: int, metadata) -> float:
    df: pd.DataFrame = pd.read_csv(f"data/lunar/training/data/S12_GradeA/{metadata['filename'].iloc[sample_ind]}.csv", parse_dates =['time_abs(%Y-%m-%dT%H:%M:%S.%f)'] ,index_col = ['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])
    df.columns = ['time', 'v']
    return df, metadata['time_rel(sec)'].iloc[sample_ind]