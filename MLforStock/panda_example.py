import pandas as pd

def test_run():
    df = pd.read_csv('data/AAPL.csv')
    print(df.head())

    print(df[10:11])

test_run()