# Neural Network From Scratch
Coding a neural network from scratch using Numpy only   
   
Pandas and Scikit-learn for data preprocessing  
    
Data used: Iris dataset from [UCI](https://archive.ics.uci.edu/ml/datasets/iris)  
  
- Backward propagation:  
    $$\frac{\partial E}{\partial X} = \frac{\partial E}{\partial Y} W^T$$  
    $$\frac{\partial E}{\partial W} = X^T\frac{\partial E}{\partial Y}$$  
    $$\frac{\partial E}{\partial B} = \frac{\partial E}{\partial Y}$$  
    $$\frac{\partial E}{\partial X} = \frac{\partial E}{\partial Y} *g'(X)$$  
  
- Loss function:  
    $$E = \frac{1}{n}\Sigma (\hat{y_i}-y_i)^2$$  
    $$\frac{\partial E}{\partial X} = \frac{2}{n}\(Y-\hat{Y})$$  
  
- Note: 
    + Delete weight files if changing network architecture  
    + One hot encoded Y is assumed  
    + Pretrained with average error of 0.000001:  
    ```
    $ python main.py 
    epoch 100/1000   error=0.000002
    epoch 200/1000   error=0.000001
    epoch 300/1000   error=0.000001
    epoch 400/1000   error=0.000001
    epoch 500/1000   error=0.000001
    epoch 600/1000   error=0.000001
    epoch 700/1000   error=0.000001
    epoch 800/1000   error=0.000001
    epoch 900/1000   error=0.000001
    epoch 1000/1000   error=0.000001
    Prediction:  [1 1 0 1 2 2 2 0 0 0]
    Actual:      [1 1 0 1 2 2 2 0 0 0]
    Accuracy:    100.0 %
    (env)
    ```  
  
- Installation:  
    + ```python -m venv env```  
    + ```pip install -r requirements.txt```  
  
- Futher optimizations:  
    + Batch, MiniBatch, Stochastic Gradient Descent
    + Batch Normalization