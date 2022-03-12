import pandas as pd


def get_data_from_csv():
    df = pd.read_csv('Coverage.csv', sep=',', header=0)
    return df


def get_columns(data, action, asm):
    # df.loc[df['column_name'] == some_value]
    rows = data.loc[(data['absolute_action'] == action) & (data['ASM'] == asm)]
    return pd.to_numeric(rows['InstructionCoverage'].str.replace(',', '.')).to_numpy()
