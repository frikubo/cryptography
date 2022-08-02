import torch as pt  #pytorch
import numpy as np

'''
- trigramy
- kluc 3x3
- text zacina na 'DRAHYJURAJ'

!DRAHYJURAJ!
"BALQTGFGYN FUHVLOIVCGPRZJUTHGWOVWCWAJGWN"
"PCPOVZOJYE JXJLVINLJMIAVAVEUKZLERO"
"NMUSMRFJGR WSWVKKDJKYTYTNSVMOJW"

'''

if __name__ == '__main__':

    with open('texty.txt', 'r') as text:
        data = text.read().replace('\n', '').split(',')

    device = 'cuda'
    y_matrices = pt.tensor([[ord(t[c]) - 65 for c in range(9)] for t in data]).to(device)
    y_matrices = y_matrices.reshape((3, 3, 3)).permute((0, 2, 1))

    x_matrix = pt.tensor([ord(c) - 65 for c in 'DRAHYJURA'], dtype=float).to(device)
    x_matrix = x_matrix.reshape((3, 3)).permute((1, 0))

    print(x_matrix)
    xm_inverse = pt.inverse(x_matrix)
    print(xm_inverse)
    