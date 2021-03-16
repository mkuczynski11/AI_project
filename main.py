import pandas as pd
import numpy as np
import functions as f

def main():
    data = pd.read_csv("data.csv")
    data = data[['song_name','genre','tempo']]
    print(data)
if __name__ == '__main__':
    main()