# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:20:17 2019

@author: Andrei


    DPModel     Model  Workers   Eps
0  0.135650  0.988433        2  0.01
3  0.296983  0.989917        7  0.01
6  0.390350  0.991200       15  0.01
1  0.391533  0.988250        2  0.10
4  0.928200  0.990733        7  0.10
7  0.958133  0.991417       15  0.10
8  0.967183  0.990900       15  0.80
2  0.971417  0.988233        2  0.80
5  0.977867  0.988900        7  0.80


    DPModel     Model  Workers   Eps
   0.630767  0.990167       30  0.01
   0.731600  0.990883       80  0.01
   0.739117  0.989600       40  0.01

"""
import torch as th
import numpy as np
import pandas as pd
from syft.frameworks.torch.differential_privacy import pate
import torchvision.datasets as datasets


device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

class Flatten(th.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SimpleCNN(th.nn.Module):
  def __init__(self, model_id, shape, 
               convs=[
                       (16,5,2), 
                       (32,3,2),
                       (64,3,1)],
               activ='leaky'
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
    self.all_layers.append(th.nn.Linear(conv_out_size, 256))
    self.all_layers.append(activ_func())
    self.all_layers.append(th.nn.Dropout(0.5))
    self.all_layers.append(th.nn.Linear(256, 10))
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
  loss_func = th.nn.CrossEntropyLoss()
  opt = th.optim.Adam(params=model.parameters())
  if verbose:
    print("Starting training worker {} on {}".format(model.model_id, _X.shape))
  for epoch in range(epochs):
    lst_loss = []
    for i_batch, (x_batch, y_batch) in enumerate(loader):
      th_output = model(x_batch)
      th_loss = loss_func(th_output, y_batch)
      opt.zero_grad()
      th_loss.backward()
      opt.step()
      lst_loss.append(th_loss.detach().cpu().numpy())
    if verbose:
      print("Worker {} Trained epoch {}, loss={:.4f}".format(
          model.model_id, epoch + 1, np.mean(lst_loss)))
    else:
      print('.', end='')
  if verbose:
    print("Worker {} done.".format(model.model_id))
  else:
    print()
  model.eval()
  return model


def add_noise_to_preds(preds, eps=0.1):
  beta = 1 / eps
  preds = preds + np.random.laplace(0, beta, preds.shape[0])
  return preds


def distribute_training_and_aggregate_results(nr_workers, X, y, eps=0.1, epochs=5):
  all_preds = []
  num_labels = y.cpu().unique().numpy().size
  ds_size = X.shape[0] // nr_workers
  nr_obs = X.shape[0]
  print("Starting training in remote locations...")
  for worker in range(nr_workers):
    model = SimpleCNN(worker +1, X.shape[1:]).to(device)
    #assume each worker has own dataset
    w_start = worker * ds_size
    w_end = min((worker + 1) * ds_size, X.shape[0])
    x_worker = X[w_start:w_end]
    y_worker = y[w_start:w_end]
    worker_model = train_pred_worker(x_worker, 
                                     y_worker, model, epochs=epochs)
    print("Using worker {} to generate predictions on {}".format(worker + 1, X.shape))
    w_preds = worker_model(X).detach().cpu().numpy()
    w_preds = np.argmax(w_preds, axis=1)
    all_preds.append(w_preds)
    del worker_model
  
  np_preds = np.array(all_preds) # WxM
  print("Combined predictions: {}".format(np_preds.shape))
  # all ctions done
  y_train = []
  print("Performing DP query and adding laplace noise...")
  for _obs in range(nr_obs):
    np_obs = np_preds[:,_obs]
    label_counts = np.bincount(np_obs, minlength=num_labels)
    label_counts = add_noise_to_preds(label_counts, eps=eps)
    y_train.append(np.argmax(label_counts))
  print("Labels ready, training final model...")
  y_train = th.tensor(np.array(y_train)).to(device)
  model = SimpleCNN(100, shape=X.shape[1:]).to(device)
  # now assume we do not have y but we just generated
  model  = train_pred_worker(X, y_train, model, epochs=epochs)
  del y_train
  print("Done training local model.")
  return model
  

def test_model(model, X_test, y_test):
  model.eval()
  yhat = model(X_test).detach().cpu().numpy()
  yhat = np.argmax(yhat, axis=1)
  acc = (yhat == y_test).sum() / (y_test.shape[0])
  print("Model {} accuracy: {:.1f}%".format(model.model_id, acc * 100))
  return acc
  

if __name__ == '__main__':
  
  SIMPLE_TESTS = False
  
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
  
  mnist_trainset = datasets.MNIST(root='d:/data', train=True, download=True, transform=None)  
  
  X_train = (mnist_trainset.train_data.unsqueeze(1).float() / 255).to(device)
  y_train = mnist_trainset.train_labels.to(device)

  np_idxs = np.arange(X_train.shape[0])
  np.random.shuffle(np_idxs)
  X_train = X_train[np_idxs]
  y_train = y_train[np_idxs]

  X_test = (mnist_trainset.test_data.unsqueeze(1).float() / 255).to(device)
  y_test = mnist_trainset.test_labels.cpu().numpy()
  
  res = {
      'DPModel': [],
      'Model': [],
      'Workers': [],
      'Eps' : [],
      }
  epochs = 5
  for nr_workers in [40, 80]: # 30, 2, 7, 15]:
    for eps in [0.01]: #, 0.1, 0.8]:
      final_model = distribute_training_and_aggregate_results(
          nr_workers=nr_workers, 
          X=X_train,
          y=y_train,
          eps=eps,
          epochs=epochs)
      print("Final Differential Privacy result:")
      acc_dpm = test_model(final_model, X_test, y_test)
      
      print("Training simple model:")
      classic_model = SimpleCNN(200, shape=X_train.shape[1:]).to(device)
      classic_model = train_pred_worker(X_train, y_train, classic_model)
      acc_mod = test_model(classic_model, X_test, y_test)
      del final_model
      del classic_model
      res['DPModel'].append(acc_dpm)
      res['Model'].append(acc_mod)
      res['Workers'].append(nr_workers)
      res['Eps'].append(eps)

  print("Model:\n{}".format(SimpleCNN(0, shape=X_train.shape[1:])))
  print(pd.DataFrame(res).sort_values('DPModel'))
  
  

  
  