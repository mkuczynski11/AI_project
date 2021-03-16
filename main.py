import pandas as pd
import numpy as np
import functions as f

def main():
    data = pd.read_csv("data.csv")
                                        #data['artists']    -   artists
                                        #data['name']       -   titles
                                        #data['year']       -   realease year
    artists = f.read_artists(data['artists'])    #list, that consists lists of strings, whcich are the authors of the song
if __name__ == '__main__':
    main()