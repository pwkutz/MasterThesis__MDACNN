import sys

import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.initializers import glorot_uniform
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


class MDACNN:
    def __init__(self, stack):
        self.model = self.mdacnn(stack) # defined but NN model
        self.predictions = None



    def mdacnn(self, stack):
        X_input = layers.Input(stack.InputShape)

        #Stage 1
        # amount kernels == 64, size kernel == (3,4), stride kernel == 1
        # X = layers.Conv2D(64, (3, 4), stride=1, name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
        X = layers.Conv2D(64, (stack.KernelHeight, stack.KernelWidth),
                          strides=1, name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)

        # Switch CONV->FC
        X = layers.Flatten()(X)

        #Stage 2.1
        # linear behaviour == linear activation function (identity function)
        # activation function not specifically defined == activation function is linear.
        # Dense(1) ... one neuron == one output
        X1 = layers.Dense(1)(X)

        #Stage 2.2
        # non-linear behaviour == non-linear activation function (tanh)
        X2 = layers.Dense(10, activation='tanh')(X)  # #1 layer
        X2 = layers.Dense(10, activation='tanh')(X2)  # #2 layer

        # used for benchmark "Continuous oscillation functions with nonlinear relationship"
        #X2 = layers.Dense(32, activation='tanh')(X) # #1 layer
        #X2 = layers.Dense(32, activation='tanh')(X2) # #2 layer
        #X2 = layers.Dense(32, activation='tanh')(X2) # #2 layer


        X2 = layers.Dense(1)(X2) # #3 layer == linear activation.

        #Stage 3
        X = layers.Add()([X1, X2]) # add together
        X = layers.Dense(1)(X) # Output Layer

        model = Model(inputs=X_input, outputs=X, name='MDA_CNN')

        return model


    def launch_mdacnn(self, stack, trainset):

        stack.training = True # needed to plot learning results on training data

        # hyperparameters for training
        self.model.compile(loss='mse',
                          optimizer='adam',
                          metrics=['mse'])

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=stack.checkpoint_path,
                                                         monitor='val_accuracy',
                                                         mode='max',
                                                         save_weights_only=True,
                                                         verbose=1)
        # training
        print("trainset.TrainDataset.shape:", trainset.TrainDataset.shape)
        print("stack.InputShape:", stack.InputShape)
        self.model.fit(trainset.TrainDataset, # Features
                          trainset.HFdataset, # force x component
                          epochs=stack.epochs, # Number of epochs
                          verbose=1,
                          batch_size=stack.BatchSize, #,
                          # validation_data=(test_features, test_target
                          callbacks=[cp_callback])


    def load_model(self, stack):
        # self.history = load_model('mdacnn_trained')
        self.model.load_weights(stack.checkpoint_path)

    def predicts(self, dataset):
        predictions = self.model.predict(dataset)  # prediction
        self.predictions = [k for i in predictions.tolist() for k in i]

    def de_normalize(self, RandomLocations, RandomLocations__scaler, HFdataset, HFdataset__scaler):

        '''
        De-normalize aka. rescale test data after propagation (before propagation it got normalized).
        For visualization.
        '''

        RandomLocations = de_normalize(RandomLocations, RandomLocations__scaler)
        self.predictions = de_normalize(np.array(self.predictions).reshape(-1, 1), HFdataset__scaler)
        HFdataset = de_normalize(HFdataset.numpy(), HFdataset__scaler)

        HFdataset = tf.convert_to_tensor(HFdataset)
        self.predictions = [x for sub in self.predictions.tolist() for x in sub]

        return RandomLocations, HFdataset





    def plot__experimentAF(self, stack, trainset):


        if not stack.experimentG:
            (trainset.RandomLocationsTest,
             trainset.HFdatasetTest) = self.de_normalize(trainset.RandomLocationsTest, trainset.RandomLocationsTest__scaler,
                                                    trainset.HFdatasetTest, trainset.HFdatasetTest__scaler)


        Y_LF = trainset.LFfunction(trainset.RandomLocationsTest[-stack.amount_HFsamples:], stack)
        Y_HF = trainset.HFdatasetTest
        Y_P = self.predictions
        X = list(trainset.RandomLocationsTest[-stack.amount_HFsamples:])
        Y_LF = [x for _, x in sorted(zip(X, Y_LF))]  # sort X and sort Y_LF accordingly
        Y_HF = [x for _, x in sorted(zip(X, list(Y_HF.numpy())))]  # sort X and sort Y_LF accordingly
        Y_P = [x for _, x in sorted(zip(X, Y_P))]
        X.sort()

        accuracy = mean_squared_error(list(trainset.HFdatasetTest.numpy()), self.predictions)

        #listH = [Y_HF, Y_LF, Y_P]
        #listHF = [k for i in listH for k in i]

        #Y_min = min(listHF)
        #Y_max = max(listHF)

        #X_max = max(np.array(trainset.RandomLocationsTest[-stack.amount_HFsamples:]))
        #X_min = min(np.array(trainset.RandomLocationsTest[-stack.amount_HFsamples:]))

        plt.style.use("seaborn-v0_8")
        # print("style.available:", plt.style.available)
        plt.plot(X, Y_LF, color='b', marker='.', label='LF')
        plt.plot(X, Y_HF, color='r', marker='.', label='HF')
        plt.plot(X, Y_P, color='g', linestyle='--', marker='.', label='MDACNN')

        plt.title("MDACNN - Test Dataset")
        plt.xlabel("Samples")
        plt.ylabel("Value")
        plt.legend(["LF", "HF", "Approximated HF"])

        plt.grid(True)
        #plt.xlim(X_min, X_max)  # show only in range of all X values
        #plt.ylim(Y_min, Y_max)  # show only range of all Y values

        #textstr = 'Accuracy=%.2f' % accuracy
        #plt.text(X_min - (np.sqrt(X_min ** 2 + X_max ** 2) / 8), Y_min + (np.sqrt(Y_min ** 2 + Y_max ** 2) / 2),
        #         textstr, fontsize=14)
        plt.subplots_adjust(left=0.25)
        plt.show()
        print("Accuracy (MSE):", accuracy)

    def plot__experimentG(self, trainset):

        from sklearn.preprocessing import scale
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(np.array(trainset.HFdatasetTest).reshape(-1,1))
        Y = scaler.fit_transform(np.array(self.predictions).reshape(-1, 1))

        plt.plot(X, Y)
        plt.title("MDACNN")
        plt.xlabel("Q_HF - Ground Truth")
        plt.ylabel("Q_HF - Prediction")
        plt.show()

    def plot__learn_results_on_training_data(self, stack, trainset):

        self.predicts(trainset.TrainDataset)
        (trainset.RandomLocations,
         trainset.HFdataset) = self.de_normalize(trainset.RandomLocations, trainset.RandomLocations__scaler,
                                                    trainset.HFdataset, trainset.HFdataset__scaler)

        Y_LF = trainset.LFfunction(trainset.RandomLocations[-stack.amount_HFsamples:], stack)
        Y_HF = trainset.HFdataset
        Y_P = self.predictions
        X = list(trainset.RandomLocations[-stack.amount_HFsamples:])
        Y_LF = [x for _, x in sorted(zip(X, Y_LF))]  # sort X and sort Y_LF accordingly
        Y_HF = [x for _, x in sorted(zip(X, list(Y_HF.numpy())))]  # sort X and sort Y_LF accordingly
        Y_P = [x for _, x in sorted(zip(X, Y_P))]
        X.sort()

        accuracy = mean_squared_error(list(trainset.HFdataset.numpy()), self.predictions)

        # listH = [Y_HF, Y_LF, Y_P]
        # listHF = [k for i in listH for k in i]

        # Y_min = min(listHF)
        # Y_max = max(listHF)

        # X_max = max(np.array(trainset.RandomLocationsTest[-stack.amount_HFsamples:]))
        # X_min = min(np.array(trainset.RandomLocationsTest[-stack.amount_HFsamples:]))

        plt.style.use("seaborn-v0_8")
        # print("style.available:", plt.style.available)
        plt.plot(X, Y_LF, color='b', marker='.', label='LF')
        plt.plot(X, Y_HF, color='r', marker='.', label='HF')
        plt.plot(X, Y_P, color='g', linestyle='--', marker='.', label='MDACNN')

        plt.title("MDACNN - Train Dataset")
        plt.xlabel("Samples")
        plt.ylabel("Value")
        plt.legend(["LF", "HF", "Approximated HF"])

        plt.grid(True)
        # plt.xlim(X_min, X_max)  # show only in range of all X values
        # plt.ylim(Y_min, Y_max)  # show only range of all Y values

        # textstr = 'Accuracy=%.2f' % accuracy
        # plt.text(X_min - (np.sqrt(X_min ** 2 + X_max ** 2) / 8), Y_min + (np.sqrt(Y_min ** 2 + Y_max ** 2) / 2),
        #         textstr, fontsize=14)
        plt.subplots_adjust(left=0.25)
        plt.show()
        print("Accuracy (MSE):", accuracy)
        return None

    def analyse(self, stack, trainset):


        if stack.experimentG: self.plot__experimentG(trainset)
        else:
            self.plot__experimentAF(stack, trainset)
            if stack.training: self.plot__learn_results_on_training_data(stack, trainset)

def de_normalize(df, scaler):
    unscaled = scaler.inverse_transform(df)
    return unscaled
