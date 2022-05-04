
from numpy import mean, sqrt, square

def rms(theData0):
    rms = sqrt(mean(square(theData0)))
    return rms