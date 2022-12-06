# Graph Recommender Systems
The focus of the project is to implement and compare the traditional Matrix Factorization (numerical linear algebra) approach and IGMC (SOTA) approach across various metrics.
The code for the IGMC model is available under the folder IGMC.
The commands used to run the train and test the model is given below

```python
  from Main import train
  Params = {'Epochs':40, 'Batch_Size':50, 'LR': 1e-3, 'LR_Decay_Step' : 20, 'LR_Decay_Value' : 10, 'Max_Nodes_Per_Hop':200}
  losses, model, test_loader, test_ratings = train(params = Params)
  test(model, test_loader, test_ratings, topK=10)
```
The description of various hyper-parameters is given below.

* Epochs - The number of iterations a model is set to train
* Batch_Size - size of the mini batch used to split the dataset (train/test)
* LR - Learning rate of the model
* LR_Decay_Step - Epoch at which learning rate will be reduced
* LR_Decay_Value - The amount by which learning rate will be reduced 
* Max_Nodes_Per_Hop - Maximum number of hops considered in extracting sub graphs

The Hyperparameter testing on Max_Nodes_Per_Hop has been performed by keeping values of others as constant.
Below is the graph depicting the relation of training loss across various Max_Nodes_Per_Hop configurations.

<img title=Comparison of training loss over max_nodes_hop" alt="Alt text" src="/images/TrainingLossVsMaxHops.png">

It can be seen that as the maximum nodes increase, loss values tends to decrease for a specific number of epochs.
  
