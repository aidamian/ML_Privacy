# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:20:17 2019

@author: Andrei


Initial simple split train in workers (data overlapping between remotes and local 
but no overlapping between remotes) then train on full with labels:
            DPModel     Model  Workers   Eps
           0.135650  0.988433        2  0.01
           0.296983  0.989917        7  0.01
           0.390350  0.991200       15  0.01
           0.391533  0.988250        2  0.10
           0.928200  0.990733        7  0.10
           0.958133  0.991417       15  0.10
           0.967183  0.990900       15  0.80
           0.971417  0.988233        2  0.80
           0.977867  0.988900        7  0.80
        
        
            DPModel     Model  Workers   Eps
           0.630767  0.990167       30  0.01
           0.731600  0.990883       80  0.01
           0.739117  0.989600       40  0.01

The more advanced tests was based on retaining 40% of data for the local model and
distributing 60% of data equaly to all remote sites. 
No data overlapping between any site.        

            DPModel     Model  Workers  Eps (NLL)
           0.835567  0.990533        7  0.1
           0.928650  0.987933       15  0.1
           
            DPModel     Model  Workers  Eps (CCE)     
           0.782233  0.995150        7  0.1
           0.925133  0.994183       15  0.1

And not with more generalized model:
  
   
"""
import torch as th
import numpy as np
import pandas as pd
from syft.frameworks.torch.differential_privacy import pate
import syft as sy
import torchvision.datasets as datasets



class Flatten(th.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SimpleCNN(th.nn.Module):
  def __init__(self, model_id, shape, 
               convs=[
                       (16,5,2), 
                       (32,3,2),
                       (64,3,1)],
               linears=[256, 64],
               activ='leaky',
               drop_rate=0.7,
               softmax=True
              ):
    super(SimpleCNN,self).__init__()
    if activ == 'leaky':
      activ_func = th.nn.LeakyReLU
    else:
      activ_func = th.nn.ReLU
    self.model_id = model_id
    self.all_layers = th.nn.ModuleList()
    prev_ch = shape[0]    
    size = shape[1]
    for _conv, _kernel, _stride in convs:
      self.all_layers.append(th.nn.Conv2d(prev_ch, 
                                     _conv, 
                                     kernel_size=_kernel, 
                                     stride=_stride))
      self.all_layers.append(th.nn.BatchNorm2d(_conv))
      self.all_layers.append(activ_func())
      size  = (size - _kernel + _stride) // _stride
      conv_out_size = (size ** 2) * _conv
      prev_ch = _conv
    
    self.all_layers.append(Flatten())
    prev_units = conv_out_size
    for _L in linears:
      self.all_layers.append(th.nn.Linear(prev_units, _L))
      self.all_layers.append(th.nn.BatchNorm1d(_L))
      self.all_layers.append(activ_func())
      self.all_layers.append(th.nn.Dropout(drop_rate))
      prev_units = _L

    self.all_layers.append(th.nn.Linear(prev_units, 10))
    if softmax:
      self.all_layers.append(th.nn.Softmax(dim=1))
    

  def forward(self, x):
    for _L in self.all_layers:
      x = _L(x)
    return x
        
        
def train_pred_worker(_X, _y, model, epochs=5, verbose=False):
  data = th.utils.data.TensorDataset(_X,_y)
  model.train()
  batch_size = 128
  loader = th.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
  if type(model.all_layers[-1]) == th.nn.Linear:
    loss_func = th.nn.NLLLoss()
    post_out = th.nn.functional.log_softmax
  else:
    loss_func = th.nn.CrossEntropyLoss()
    post_out = lambda x: x
  opt = th.optim.Adam(params=model.parameters())
  if verbose:
    print("  Starting training model {} on {}".format(
        model.model_id, _X.shape))
  print("   Losses: ", end='')
  for epoch in range(epochs):
    lst_loss = []
    for i_batch, (x_batch, y_batch) in enumerate(loader):
      th_readout = model(x_batch)
      th_log_sm = post_out(th_readout)
      th_loss = loss_func(th_log_sm, y_batch)
      opt.zero_grad()
      th_loss.backward()
      opt.step()
      lst_loss.append(th_loss.detach().cpu().numpy())
    if verbose:
      print("  Worker {} Trained epoch {}, loss={:.4f}".format(
          model.model_id, epoch + 1, np.mean(lst_loss)))
    else:
      print('[ {:>4.2f} ]'.format(np.mean(lst_loss)), end='')
  if verbose:
    print("  Worker {} done.".format(model.model_id))
  else:
    print()
  model.eval()
  return model


def add_noise_to_preds(preds, eps):
  beta = 1 / eps
  preds = preds + np.random.laplace(0, beta, preds.shape[0])
  return preds

def centralize_and_anonymize(np_preds, num_labels, epsilon, do_pate=False):
  print("Combined predictions: {}, labels: {}".format(
      np_preds.shape, np.unique(np_preds)))
  # all ctions done
  y_train = []
  n_obs = np_preds.shape[1]
  print("Performing DP query and adding laplace noise eps={} on {} observation...".format(
      epsilon, n_obs))
  for _obs in range(n_obs):
    np_obs = np_preds[:,_obs]
    label_counts = np.bincount(np_obs, minlength=num_labels)
    label_counts = add_noise_to_preds(label_counts, eps=epsilon)
    y_train.append(np.argmax(label_counts))
  y_train = np.array(y_train)
  print("  Labels DP done, new labels: {}".format(np.unique(y_train)))
  if do_pate:
    print("Performing PATE analysis on...")
    data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=np_preds, 
                                                       indices=y_train, 
                                                       noise_eps=epsilon,
                                                       delta=1e-5)
    
    print("  Data Independent Epsilon: {:.10f}".format(data_ind_eps))
    print("  Data Dependent Epsilon:   {:.10f}".format(data_dep_eps))
  return y_train
  


def distribute_training_and_aggregate_results(nr_workers, 
                                              np_X_remote, 
                                              np_y_remote,
                                              np_X_local,
                                              dev,
                                              num_labels=10,
                                              eps=0.1, 
                                              epochs=5,
                                              verbose=False,
                                              do_pate=False):
  all_preds = []
  
  remote_ds_size = np_X_remote.shape[0] // nr_workers
  
  X_remote = th.tensor(np_X_remote).to(dev)
  y_remote = th.tensor(np_y_remote).to(dev)
  X_local = th.tensor(np_X_local).to(dev)
  
      
  print("Starting training in remote locations...")
  for worker in range(nr_workers):
    #assume each worker has own dataset
    w_start = worker * remote_ds_size
    w_end = min((worker + 1) * remote_ds_size, np_X_remote.shape[0])
    x_worker = X_remote[w_start:w_end]
    y_worker = y_remote[w_start:w_end]
    worker_labels = np.unique(np_y_remote[w_start:w_end])
    print(" Training data {:>5}:{:<5} on site {} w. {} labels {}".format(
            w_start, w_end, worker + 1, worker_labels.size, worker_labels))
    worker_model = SimpleCNN(worker +1, np_X_remote.shape[1:]).to(dev)
    worker_model = train_pred_worker(x_worker, 
                                     y_worker, worker_model, 
                                     epochs=epochs)
    if verbose:
      print(" Using model {} to generate predictions on {}".format(
          worker + 1, X_local.shape))
    w_preds = worker_model(X_local).detach().cpu().numpy()
    w_preds = np.argmax(w_preds, axis=1)
    all_preds.append(w_preds)
    del worker_model
  
  np_preds = np.array(all_preds) # WxM
  y_train = centralize_and_anonymize(np_preds, 
                                     num_labels, 
                                     epsilon=eps,
                                     do_pate=do_pate)
  print("Labels ready, training final model on {}...".format(
      X_local.shape))
  y_train = th.tensor(y_train).to(dev)
  # now assume we do not have y but we just generated
  model = SimpleCNN(100, shape=np_X_local.shape[1:]).to(dev)
  model  = train_pred_worker(X_local, y_train, model, epochs=epochs)
  del y_train
  print("Done training local model.")
  return model
  

def test_model(model, np_X, np_y, dev):
  X_test = th.tensor(np_X).to(dev)
  model.eval()
  yhat = model(X_test).detach().cpu().numpy()
  yhat = np.argmax(yhat, axis=1)
  acc = (yhat == np_y).sum() / (np_y.shape[0])
  print("Model {} accuracy: {:.1f}%".format(model.model_id, acc * 100))
  return acc
  

if __name__ == '__main__':
  
  SIMPLE_TESTS = False
  
  #sy.TorchHook
  
  if SIMPLE_TESTS:
    print("Simple tests:")  
    num_teachers = 10 # we're working with 10 partner hospitals
    num_examples = 10000 # the size of OUR dataset
    num_labels = 10 # number of lablels for our classifier
    
    preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int).transpose(1,0) # fake predictions
    
    new_labels = list()
    for an_image in preds:
    
      label_counts = np.bincount(an_image, minlength=num_labels)
    
      epsilon = 0.1
      beta = 1 / epsilon
    
      for i in range(len(label_counts)):
          label_counts[i] += np.random.laplace(0, beta, 1)
    
      new_label = np.argmax(label_counts)
      
      new_labels.append(new_label)
        
    
    labels = np.array([9, 9, 3, 6, 9, 9, 9, 9, 8, 2])
    counts = np.bincount(labels, minlength=10)
    query_result = np.argmax(counts)
    print(query_result)
    
    
    num_teachers, num_examples, num_labels = (100, 100, 10)
    preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int) #fake preds
    indices = (np.random.rand(num_examples) * num_labels).astype(int) # true answers
    
    preds[:,0:10] *= 0
    
    data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, 
                                                       indices=indices, 
                                                       noise_eps=0.1,
                                                       delta=1e-5)
    
    assert data_dep_eps < data_ind_eps
    
    data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5)
    print("Data Independent Epsilon:", data_ind_eps)
    print("Data Dependent Epsilon:", data_dep_eps)
    
    
    preds[:,0:50] *= 0
    
    
    
    data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5, moments=20)
    print("Data Independent Epsilon:", data_ind_eps)
    print("Data Dependent Epsilon:", data_dep_eps)
  
  print("DL tests:")
  c_dev = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

  separate_data = True
  
  mnist_trainset = datasets.MNIST(root='d:/data', train=True, download=True, transform=None)  
  
  np_X_train = (mnist_trainset.train_data.unsqueeze(1).float() / 255).numpy()
  np_y_train = mnist_trainset.train_labels.numpy()

  np_idxs = np.arange(np_X_train.shape[0])
  np.random.shuffle(np_idxs)
  np_X_train = np_X_train[np_idxs]
  np_y_train = np_y_train[np_idxs]

  np_X_test = (mnist_trainset.test_data.unsqueeze(1).float() / 255).numpy()
  np_y_test = mnist_trainset.test_labels.cpu().numpy()
  
  if separate_data:
    # we construct different sets of data for each remotes
    # local data is different from remotes
    local_size = int(np_X_train.shape[0] * 0.4)
    np_X_loc = np_X_train[:local_size]
    np_y_loc = np_y_train[:local_size]
    np_X_rem = np_X_train[local_size:]
    np_y_rem = np_y_train[local_size:]      
  else:
    # the local data is simply the aggregated data from all remotes
    np_X_rem = np_X_train
    np_X_loc = np_X_train
    np_y_rem = np_y_train
    np_y_loc = np_y_train  
  
  res = {
      'DPModel': [],
      'Model': [],
      'Workers': [],
      'Eps' : [],
      }
  epochs = 5
  for nr_workers in [30]: # 30, 2, 7, 15
    for _eps in [0.01]: #, 0.1, 0.5, 0.8
      final_model = distribute_training_and_aggregate_results(
          nr_workers=nr_workers, 
          np_X_remote=np_X_rem,
          np_y_remote=np_y_rem,
          np_X_local=np_X_loc,
          eps=_eps,
          epochs=epochs,
          dev=c_dev,
          do_pate=True,
          )
      print("Final Differential Privacy result:")
      acc_dpm = test_model(final_model, np_X_test, np_y_test, 
                           dev=c_dev)
      
      print("Training simple model:")
      th_X_trn = th.tensor(np_X_train).to(c_dev)
      th_y_trn = th.tensor(np_y_train).to(c_dev)
      classic_model = SimpleCNN(200, shape=np_X_train.shape[1:]).to(c_dev)
      classic_model = train_pred_worker(th_X_trn, 
                                        th_y_trn, 
                                        classic_model)
      acc_mod = test_model(classic_model, np_X_test, np_y_test, dev=c_dev)
      del final_model
      del classic_model
      res['DPModel'].append(acc_dpm)
      res['Model'].append(acc_mod)
      res['Workers'].append(nr_workers)
      res['Eps'].append(_eps)

  print("Model:\n{}".format(SimpleCNN(0, shape=np_X_train.shape[1:])))
  print(pd.DataFrame(res).sort_values('DPModel'))
  
  

  
  