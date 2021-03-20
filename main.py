import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
from typing import Dict, Text

import functions as f

def main():
    data = pd.read_csv("data.csv")
    data = data[['song_name','genre','tempo','energy']]
if __name__ == '__main__':
    main()