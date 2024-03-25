import numpy as np

class Network:
    def __init__(self):
        self.layers = []

    def mse(self, y, yhat):
        return np.mean(np.power(y-yhat, 2));
    
    def dmse(self, y, yhat):
        return 2*(yhat-y)/y.shape[0]

    def add(self, layer):
        self.layers.append(layer)
    
    def predict(self, input):
        """
        Input is array of vector Xs [[x1] [x1] [x3]]
        Output is array of vector ys [[y1] [y2] [y3]]
        """
        output = None
        once = False

        self.load_weights()

        for sample in input:
            layer_output = sample
            for layer in self.layers:
                layer_output = layer.forward(layer_output)
            if not once:
                output = layer_output
                once = True
            else:
                output = np.append(output, layer_output, axis=0)
        return output
    
    def save_weights(self):
        for i in range(len(self.layers)):
            np.savetxt('./weights/weights_layer' + str(i) +'.txt', self.layers[i].weights)

    def load_weights(self):
        for i in range(len(self.layers)):
            self.layers[i].weights = np.loadtxt('./weights/weights_layer' + str(i) +'.txt')

    def train(self, X_train, y_train, epochs, lr, save_weights=True, load_weights=False):
        if load_weights:
            self.load_weights()
        for epoch in range(epochs):
            # Shuffle train data
            shuffle = np.arange(X_train.shape[0])
            np.random.shuffle(shuffle)
            X_train = X_train[shuffle]
            y_train = y_train[shuffle]

            avg_loss = 0
            for i in range(X_train.shape[0]):
                sample = X_train[i]
                for layer in self.layers:
                    sample = layer.forward(sample)
                yhat = sample
                avg_loss = avg_loss + self.mse(y_train[i], yhat)

                dy = self.dmse(y_train[i], yhat)
                # Replug error:
                for layer in reversed(self.layers):
                    dy = layer.backward(dy, lr)
            
            if (epoch+1) % (epochs/10) == 0:
                print('epoch %d/%d   error=%f' % (epoch+1, epochs, avg_loss/X_train.shape[0]))
        if save_weights:
            self.save_weights()