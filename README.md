# The Choquet Integral Neural Network
This is a PyTorch implementation of the paper, "[Enabling Explainable Fusion in Deep Learning with Fuzzy Integral Neural Networks](https://arxiv.org/pdf/1905.04394.pdf)".  If you use this work, please cite as 

    @article{islam2019Enabling,
        author       = {Muhammad Aminul Islam and Derek T. Anderson and Timothy C. Havens and Grant Scott and James M. Keller},
        title        = {{Enabling Explainable Fusion in Deep Learning with Fuzzy Integral Neural Networks}},
        journal={IEEE Transactions on Fuzzy Systems},
        year         = 2019,
        doi          = {10.1109/TFUZZ.2019.2917124},
        publisher    = {IEEE},
        }

This repo consists of the Choquet Integral Neuron module described in the paper and an example that illustrates learning of the Choquet integral from a synthetic dataset. 

## Installation/Dependencies
The code uses [PyTorch](https://pytorch.org/) deep learning frameworks. So, if you haven't it installed on your system, please follow the instructions [here](https://pytorch.org/get-started/locally/). We recommend anaconda as a package manager, which takes care of all dependencies.

After installing pytorch and all its dependencies, run the following commands to download and run the example.
```
$ git clone https://github.com/aminb99/choquet-integral-NN.git
$ cd choquet-integral-NN
$ python Choquet_integral_nn_torch.py
```

## Example
The Choquet_integral_nn_torch.py includes an example at the bottom (after if __name__=="__main__":) that shows how to learn the Choquet integrals for multilabel tasks.

1. Suppose there are N_in=3 inputs and N_out=2 outputs. First, create a synthetic dataset with M samples via random sampling from a normal distribution with mean =-1 and std=2

```
    M = 700
    N_in = 3
    N_out = 2  
    X_train = np.random.rand(M,N_in)*2-1
```

2. Let's specify the FMs  (There will be N_out number of FMs). Herein we adopt binary encoding instead of lexicographic encoding to represent a FM that is easier to code. As for example, an FM for three inputs using lexicographic encoding is, g = {g_1, g_2, g_3, g_{12}, g_{13}, g_{23}, g_{123}} whereas its binary encoding is g = {g_1, g_2, g_{12}, g_3 g_{13}, g_{23}, g_{123}}.

For simplicity, here we use OWA. 
```
    OWA = np.array([[0.7, 0.2, 0.1], # this is soft-max
                    [0.1,0.2,0.7]])  # soft-min
```
The FMs of the above OWAs in binary encoding
```
   FM = [[0.7, 0.7, 0.9, 0.7, 0.9, 0.9, 1.0].
       [0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 1.0]]
```
3. Given these FMs, compute the label or graoundtruth for the training dataset.
```
    label_train = np.matmul(np.sort(X_train), np.fliplr(OWA).T)
```
4. Create a NN with two output neurons.
```
    net = Choquet_integral(N_in,N_out)
```
5. Optimize
```
    # set the optimization algorithms and paramters the learning
    learning_rate = 0.3;
    
    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)   
    
    num_epochs = 300;
    
    # convert from numpy to torch tensor
    X_train = torch.tensor(X_train,dtype=torch.float)
    label_train = torch.tensor(label_train,dtype=torch.float)
    
    # optimize
    for t in range(num_epochs):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = net(X_train)
    
        # Compute the loss
        loss = criterion(y_pred, label_train)
    
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad())
        loss.backward()
        optimizer.step()  
   ```
6. Finally the learned FMs
```
 FM_learned = (net.chi_nn_vars(net.vars).cpu()).detach().numpy()
```


## Notes
This implementation differs from the article in the respect that we used sort operations to compute the coefficients whereas the article uses a network termed coefficient network that employs max and min and does not require sorting operation.   



