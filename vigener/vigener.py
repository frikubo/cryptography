from kassisk import kassisk
from decoder import decodeAll
import helpers as h
import numpy as np

splitchar = '%'

if __name__ == '__main__':
    # otvori suboor s vstupnymi textami oddelenymi %
    with open('texty.txt', 'r') as text:
        data = text.read().split(splitchar)

    # filter textu - len samotne pismena
    formated = h.onlyChars(data)

    # vysledky kassiskeho rozboru jednotlivych textov

    print('Kassisky\'s test running...')
    results = kassisk(formated)
    # mozne dlzky kluca <20, 29>
    # r ako Tuple
    print('Calculating possible lengths of keys...')
    keyLengths = h.findPossibleKeyLengths(results, (20, 29))

    print('Done, generating KEYS ...')
    # generujeme kluc
    k = h.findKeys(formated, keyLengths)

    print('Decoding text...')

    # vsetko dekoduje
    original = ''

    # slovenske
    s = np.array(k)[:, 0].tolist()
    original += 'SLOVAK \n'
    original = decodeAll(data, s, original)

    # anglicke
    e = np.array(k)[:, 1].tolist()
    original += 'ENGLISH \n'
    original = decodeAll(data, e, original)

    # ulozi do suboru
    with open('decoded.txt', 'w') as decoded:
        decoded.write(original)
    print('Done. Original text written to file.')