# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:09:24 2019

@author: Andrei
"""


import torch

def get_parallel_db(db, remove_index):
  return torch.cat((db[0:remove_index], db[remove_index+1:])) 


def get_parallel_dbs(db):
  parallel_dbs = list()
  for i in range(len(db)):
    pdb = get_parallel_db(db, i)
    parallel_dbs.append(pdb)  
  return parallel_dbs


def create_db_and_parallels(num_entries):  
  db = torch.rand(num_entries) > 0.5
  pdbs = get_parallel_dbs(db)  
  return db, pdbs

def basic_query(db):
  return torch.mean(db.float())
  

def randomized_response(db, thr=0.5):
  true_result = basic_query(db)
  first_coin_flip = (torch.rand(len(db)) > thr).float()
  second_coin_flip = (torch.rand(len(db)) > 0.5).float()
  # now we augment the data BEFORE it arrives in our pipeline  
  augmented_db = db.float() * first_coin_flip + (1 - first_coin_flip) * second_coin_flip
  # the augmented data enters the pipeline
  augmented_result = basic_query(augmented_db)
  # now we have to compute the "backward morphism"
  sk_result = ((augmented_result / thr) - 0.5) * thr / (1 - thr)
  return sk_result, true_result


if __name__ == '__main__':
  for _len in [10, 100, 1000, 10000]:
    for _thr in [0.1, 0.4, 0.8]:
      _db, _pdbs = create_db_and_parallels(_len)
      _ar, _tr = randomized_response(_db, _thr)
      print("DB[{:>7}] Rand: {:.1f} Augmented: {:.3f} vs True: {:.3f} (diff={:.4f})".format(
          _len, _thr, _ar, _tr, abs(_tr-_ar)))
  



