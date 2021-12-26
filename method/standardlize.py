import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def standardlize(df):
    values = df.values
    values = values.astype(np.float)
    scaler = StandardScaler()
    df_std = scaler.fit_transform(values)
    df_std = pd.DataFrame(df_std, index=df.index, columns=df.columns)
    mean , var = np.array(scaler.mean_) , np.array(scaler.var_)
    np.savetxt('mean.txt', mean)
    np.savetxt('var.txt', var)
    return df_std