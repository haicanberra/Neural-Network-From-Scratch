# NumPy Neural Network
## About
A neural network written in Python from scratch using NumPy module only, able to save and load weights.

## Packages
NumPy, Pandas, Scikit-learn  

## References
Iris dataset from [UCI](https://archive.ics.uci.edu/ml/datasets/iris)  

## Mathematics
- Backward propagation:  
    $$\frac{\partial E}{\partial X} = \frac{\partial E}{\partial Y} W^T$$  
    $$\frac{\partial E}{\partial W} = X^T\frac{\partial E}{\partial Y}$$  
    $$\frac{\partial E}{\partial B} = \frac{\partial E}{\partial Y}$$  
    $$\frac{\partial E}{\partial X} = \frac{\partial E}{\partial Y} *g'(X)$$ 
- Loss function:  
    $$E = \frac{1}{n}\Sigma (\hat{y_i}-y_i)^2$$  
    $$\frac{\partial E}{\partial X} = \frac{2}{n}(Y-\hat{Y})$$  
  
## Notes 
- Delete weight files in weights folder if changing network architecture (in main.py).
- One hot encoded Y is assumed: [[y1][y2]...[yn]] for n samples.
- Pretrained with Mean Squared Error of $10^{-6}$:  
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
  
## Installation:  
```
python -m venv env
source env/Scripts/activate
pip install -r requirements.txt
```  
  
## Futher Optimizations:  
- Batch, MiniBatch, Stochastic Gradient Descent.
- Batch Normalization.
