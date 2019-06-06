# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:09:24 2019

@author: Andrei
"""

import numpy as np
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

def sensitivity(query_func, db):
  pdbs = get_parallel_dbs(db)
  full_db_result = query_func(db)
  sensitivity = 0
  for pdb in pdbs:
    pdb_result = query_func(pdb)    
    db_distance = torch.abs(pdb_result - full_db_result)    
    if(db_distance > sensitivity):
        sensitivity = db_distance  
  return sensitivity

def mean_query(db):
  return torch.mean(db.float())

def sum_query(db):
  return torch.sum(db.float())

def laplace_noise(db, query_func, eps):
  """
  eps controls the ammount of noise: the smaller the smaller the 
  ammount of leaked information and thus the bigger the ammount of noise
  """
  b = sensitivity(query_func, db) / eps
  print(" Beta: {:.4f}".format(b))
  noise = np.random.laplace(loc=0.0, scale=b)
  return noise
  
def M(db, query_func, noise_func, eps):
  """
  generic global differential privacy algorithm
  """
  return query_func(db) + torch.tensor(noise_func(db, query_func, eps))


if __name__ == '__main__':
  eps = 0.5
  dbs = []
  
  for _len in [10, 100, 1000, 10000]:    
    print()
    _db, _ = create_db_and_parallels(_len)
    dbs.append(_db)
    _ms = sensitivity(mean_query, _db)
    _ss = sensitivity(sum_query, _db)
    print("Mean sensitivity for DB[{:>5}]: {:.5f}".format(_len, _ms))
    print("Sum  sensitivity for DB[{:>5}]: {:.5f}".format(_len, _ss))
    
    dbrm_gdb = M(_db, mean_query, laplace_noise, eps=eps)
    dbrs_gdb = M(_db, sum_query, laplace_noise, eps=eps)
    print(" GD mean result: {:.4f}  vs Real result: {:.4f}".format(
        dbrm_gdb, mean_query(_db)))
    print(" GD sum  result: {:.4f}  vs Real result: {:.4f}".format(
        dbrs_gdb, sum_query(_db)))

  
  
  
  
  
  
  
