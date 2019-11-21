#Normalizar campos max min
def normalizacionColumna(df, i):
    columns = df.columns.values
    df[columns[i]] = (df[columns[i]] - df[columns[i]].min()) / (df[columns[i]].max() - df[columns[i]].min())

def normalizarDataset(dataset, indices):
    df = dataset.copy()
    for i in indices:
        normalizacionColumna(df, i)
    return df

def normalize_columns(index,df):
    data = normalizarDataset(df, index)
    return data