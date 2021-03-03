#file to store functions 
def readArtists(pnArtists) -> list[list[str]]:          #function accepts list[str] and especially is designed to handle pandas.core.series.Seres() object
    artists:list[list[str]] = []                        #every string is translated meaning:
    for x in pnArtists:                                 #"'Artist'" , " 'Artists2", " 'Artists3"
        artistsFixed = x[1:-1]                          #is translated to be
        artistsFixed = artistsFixed.split(',')          #"Artist" , "Artist2", "Artist3"
        artistsToAppend:list[str] = []                  #and these values are stored in list[str], whcich at the end gets into new artist list which we return
        for a in artistsFixed:
            s:str = ""
            if a[0] == '\'':
                s = a[1:-1]
            if a[0] == ' ':
                s = a[2:-1]
            artistsToAppend.append(s)
        artists.append(artistsToAppend)
    return artists