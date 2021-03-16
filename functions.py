import pandas as pd
import numpy as np

#file to store functions 
def read_artists(pn_artists: pd.DataFrame) -> list[list[str]]:        #function accepts list[str] and especially is designed to handle pandas.core.series.Seres() object
    artists:list[list[str]] = []                        #every string is translated meaning:
    for x in pn_artists:                                #"'Artist'" , " 'Artists2'", " 'Artists3'"
        i=1                                             #is translated to be
        if(x[0] == '"'):                                #"Artist" , "Artist2", "Artist3"
            while(x[i] != '"' or x[i] != "'"):          #and these values are stored in list[str], whcich at the end gets into new artist list which we return
                i+=1
        artists_fixed = x[i:-i]
        artists_fixed = artists_fixed.split(',')          
        artists_to_append:list[str] = []                  
        for a in artists_fixed:
            s:str = ""
            if a[0] == '\'' or a[0] == '"':
                s = a[1:-1]
            if a[0] == ' ':
                s = a[2:-1]
            artists_to_append.append(s)
        artists.append(artists_to_append)
    return artists