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
- Installation: ```pip install -r requirements.txt```