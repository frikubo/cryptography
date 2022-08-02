import torch
import numpy as np
import re
import datetime

from torch.multiprocessing import Process, Pipe
from torch.multiprocessing import set_start_method

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

def worker(id, path, c, iterations, input, device):
  input = input.to(device)
  seeds = []
  for it in range(iterations):
    trand = torch.floor(c.recv().to(device) * 26).narrow(0, 0, length=input.shape[0])

    res = torch.remainder((input - trand) + 26, 26)

    binc = torch.bincount(res.type(dtype=torch.int32), minlength=26)
    index = torch.sum((binc / res.shape[0]) * ((binc - 1) / (res.shape[0] - 1)))

    if index > 0.055:
      seeds.append(it)

  np.savetxt(f'{path}text{id}_seeds.txt', np.array(seeds), delimiter='\n')
  c.recv()
  c.close()

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f'device: {device}')
  set_start_method('spawn')

  path = ''

  iterations = 217728
  texts_num = 4

  print('Initializing')
 
  processes, connections = [], []
  max_text_len = 1e-9
  for id in range(texts_num):
    parent_connection, child_connection = Pipe()

    with open(f'{path}text{id + 1}_enc.txt', 'r') as text:
        data = re.sub("[^A-Z]", "", text.read())
        max_text_len = max(max_text_len, len(data))

    process = Process(target = worker, args = (id + 1, path, child_connection, iterations, torch.tensor([ord(i) - 65 for i in data]), device))
    connections.append(parent_connection)
    processes.append(process)
    process.start()

  g = Generator()

  print('Processing')
  start = datetime.datetime.now()

  for it in range(iterations):
    if(it % 10000 == 0 and it != 0):
        print(f'{it + 1}. iteration\'s done')
    g.seed(it)
    x = g.next_tensor(max_text_len, 'cpu')
    for c in connections:
      c.send(x)
  
  [connection.send(1) for connection in connections]
  [process.join() for process in processes]

  print('Done ~ ', datetime.datetime.now() - start)