import os
import pprint
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
from typing import Dict, Text

import functions as f

def main():
    ds = tfds.load('movielens/latest-small-ratings', split='train')

    ds = ds.take(20).cache()
    print(ds)

    for example in ds:
        print(list(example.keys()))
        title = example['movie_title']
        rating = example['user_rating']
        print(f'{title} - {rating}')


if __name__ == '__main__':
    main()