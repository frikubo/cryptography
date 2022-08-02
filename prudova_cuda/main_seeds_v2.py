import torch
import numpy as np
import re
import datetime

from torch.multiprocessing import Process, Pipe
from torch.multiprocessing import set_start_method

class Generator():
  def __init__(self, device, seed = 0):
    self.device = device
    self.init_seed = seed
    self.x = torch.tensor(np.array([seed])).to(device)
    self.seeds_tensor = None

  def next_rand(self):
    self.x = (84589 * self.x + 45989) % 217728
    return self.x / 217728.0

  def next_rand_tensor(self):
    self.seeds_tensor = (84589 * self.seeds_tensor + 45989) % 217728
    return (self.seeds_tensor / 217728.0).unsqueeze(1)

  def get_seed(self):
    return self.init_seed

  def seed(self, seed):
    self.init_seed = seed
    self.x[0] = seed

  def next_tensor(self, size):
    return torch.tensor(np.array([self.next_rand() for _ in range(size)]), device=self.device, dtype=float).unsqueeze(1)

  def next_batch(self, size, start_seed, batch_size):
    self.seeds_tensor = torch.tensor(np.array([s for s in range(start_seed, start_seed + batch_size)])).to(self.device)
    res = torch.Tensor().to(self.device)
    for _ in range(size):
        res = torch.cat((res, self.next_rand_tensor()), dim=-1)
    return res.transpose(1, 0)


def worker(id, path, c, batch_size, iterations, max_text_len, input, device):
  input = input.to(device).unsqueeze(1)
  input = torch.repeat_interleave(input, batch_size, 1)
  seeds = torch.Tensor().to(device)
  running = True
  g = Generator(device)
  steps = 0

  while(running):
    size = batch_size if steps + batch_size <= iterations else iterations - steps
    batch = g.next_batch(max_text_len, steps, size)

    if steps + batch_size >= iterations:
        running = False

    steps += size
    if id == 1:
        print(steps)
    trand = torch.floor(batch * 26).narrow(0, 0, length=input.shape[0])

    diff = (input if trand.shape[1] == batch_size else input.narrow(1, 0, length=trand.shape[1])) - trand
    res = torch.remainder(diff + 26, 26)

    bincs = torch.Tensor().to(device)
    for c in range(res.shape[1]):
        bincs = torch.cat((bincs, torch.bincount(res[:, c].type(dtype=torch.int32), minlength=26).unsqueeze(1)), dim=1)

    '''
    bincs = torch.tensor(np.array([torch.bincount(res[:, c].type(dtype=torch.int32), minlength=26).cpu().numpy() for c in range(res.shape[1])]), 
                            device=device).transpose(1, 0)
    '''
    indexes = torch.sum((bincs / res.shape[0]) * ((bincs - 1) / (res.shape[0] - 1)), (0,))
    seeds = torch.cat(((indexes > 0.055).nonzero(as_tuple=False) + steps, seeds))

  np.savetxt(f'{path}text{id}_seeds.txt', seeds.cpu().numpy(), delimiter='\n')
  c.recv()
  c.close()

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  device = 'cpu'
  print(f'device: {device}')
  set_start_method('spawn')

  path = ''
  batch_size = 4096

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

    process = Process(target = worker, 
                        args = (id + 1,
                                path, 
                                child_connection, 
                                batch_size, 
                                iterations, 
                                max_text_len, 
                                torch.tensor(np.array([ord(i) - 65 for i in data])), 
                                device))

    connections.append(parent_connection)
    processes.append(process)
    process.start()

  print('Processing')
  start = datetime.datetime.now()
  
  [connection.send(1) for connection in connections]
  [process.join() for process in processes]

  print('Done ~ ', datetime.datetime.now() - start)