import pandas as pn
import functions as f

def main():
    data = pn.read_csv("data.csv")
                                        #data['artists']    -   artists
                                        #data['name']       -   titles
                                        #data['year']       -   realease year
    artists = f.read_artists(data['artists'])    #list, that consists lists of strings, whcich are the authors of the song

if __name__ == '__main__':
    main()