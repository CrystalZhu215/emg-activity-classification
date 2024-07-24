import pandas as pd

train_dfs, test_dfs = [], []

def extract_data(group : str, batch_size : int) -> None:
    """
    Saves extracted data as csv in data/train or data/test.

    Parameters
    ----------
    group : str
        'train' or 'test
    batch_size : int
        number of samples to extract per subject, repetition number, and label
    """

    samples = pd.DataFrame()

    for i in range(1, 26):
        if i < 10:
            num_code = '0' + str(i)
        else:
            num_code = str(i)

        df = pd.read_csv(f'data_all/{group[0].upper() + group[1:]}CSV_C23/S0{num_code}_{'tr' if group == 'train' else 'tt'}.csv')
        
        for r in pd.unique(df['Repition']):
            for l in pd.unique(df['Label']):
                rows = df[(df['Repition'] == r) & (df['Label'] == l)].iloc[:batch_size]
                samples = pd.concat([samples, rows], axis=0, ignore_index=True)

    samples.to_csv(f'data/{group}/{group}_data.csv')

print('Extracting train data...')
extract_data('train', 1000)

print('Extracting test data...')
extract_data('test', 10)