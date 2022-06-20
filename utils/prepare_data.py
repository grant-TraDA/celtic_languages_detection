import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(file):
    data_all = pd.read_csv(file, sep='\t', header=None, quoting=3, error_bad_lines=False)

    data_all[0] = data_all[0].astype(str)
    data_all[1] = data_all[1].astype("category").cat.codes

    data = data_all.iloc[:, 0]
    target = data_all.iloc[:, 1]

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    return x_train, y_train, x_test, y_test, data
