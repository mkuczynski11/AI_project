import pandas as pn
import functions as f
data = pn.read_csv("data.csv")
                                    #data['artists']    -   artists
                                    #data['name']       -   titles
                                    #data['year']       -   realease year
artists = f.readArtists(data['artists'])    #list, that consists lists of strings, whcich are the authors of the song