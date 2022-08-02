import torch
import numpy as np
import re

class Generator():
  def __init__(self, seed = 0):
    self.init_seed = seed
    self.x = seed

  def next_rand(self):
    self.x = (84589 * self.x + 45989) % 217728
    return self.x / 217728.0

  def get_seed(self):
    return self.init_seed

  def seed(self, seed):
    self.init_seed = seed
    self.x = seed

  def next_tensor(self, size, device):
    return torch.tensor([self.next_rand() for _ in range(size)], device=device)

'''
text1: 77777
'''

if __name__ == '__main__':
    path = ''
    id = 4

    with open(f'{path}text{id}_enc.txt', 'r') as text:
        enc = text.read()

    with open(f'{path}text{id}_seeds.txt', 'r') as text:
        seeds = text.read().split('\n')

    g = Generator()

    for seed in seeds:
        g.seed(int(float(seed)))
        dec = ''
        for c in enc:
            if ord(c) - 65 >= 0 and ord(c) - 65 < 26:
                dec += chr((ord(c) - 65 + 26 - int(g.next_rand() * 26)) % 26 + 65)
            else:
                dec += c
        print(dec)
        print(f'seed: {seed}')
        input("Press Enter to continue...") 



