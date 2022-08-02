import re
import numpy as np
import math
import copy
import concurrent.futures as cf
from itertools import repeat
import enums.english as e, enums.slovak as s

# pravdepodobnost vyskytu jednotlivych znakov vybranych monoalfabetickych retazcov v texte
def calculateProbs(text, keyLength):
    probs = [None] * keyLength
    # pocet stlpcov
    for i in range(keyLength):
        count = 0
        # znaky v stlpci
        for c in range(i, len(text), keyLength):
            count += 1
            if probs[i] is None:
                probs[i] = [0]*26
            probs[i][ord(text[c]) - 65] += 1
        probs[i] = (np.array(probs[i]) / count).tolist()
    return probs

#c ako character, kc ako 0..25
def decodeChar(c, kc):
    return chr(((ord(c) - 65) + 26 - kc) % 26 + 65)

def generateKeys(text, keyLength, results, id):
    probs = calculateProbs(text, keyLength)
    key = np.char.array(['', ''])
    #postupne sa generuje znak po znaku
    for i in range(keyLength):
        #skusa sa od A po Z
        res = np.zeros((26, 2))
        for t in range(26):
            # sum error pre jednotlive znaky
            errsum = [0, 0]
            # vypocet sumy chyb pre anglicku aj slovensku abecedu
            for c in range(i, len(text), keyLength):
                errsum[0] += math.pow( probs[i][ord(text[c]) - 65] - eval('s.Slovak.' + decodeChar(text[c], t)), 2)
                errsum[1] += math.pow( probs[i][ord(text[c]) - 65] - eval('e.English.' + decodeChar(text[c], t)), 2)
            #vyber minimalnu
            res[t, :] = errsum
        # vybere sa index s minimalnou chybou -> znak v kluci
        key = key + np.char.array([chr(np.argmin(res[:, 0]) + 65), chr(np.argmin(res[:, 1]) + 65)])
    results[id] = key

# pokusi sa najst kluce
def findKeys(data, keyLengths):
    results = [None] * len(keyLengths)
    for t in range(len(keyLengths)):
        generateKeys(data[t], keyLengths[t], results, t)
    return results

# Ostanu len pismena A-Z
def onlyChars(data):
    res = copy.deepcopy(data)
    for t in range(len(res)):
        res[t] = re.sub("[^A-Z]", "", res[t])
    return res

def findPossibleKeyLengths(data, r):
    res = [int] * len(data)
    for d in range(len(data)):
        max = -1
        k = 0
        #20 - 29
        for i in range(r[0], r[1] + 1):
            c = 0
            for t in data[d] :
                if int(t[0]) % i == 0 :
                    c += 1
            
            if c > max:
                max = c
                k = i
        res[d] = k
    
    return res
