import os, sys

import keras
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from tensorflow.keras import layers, models, Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import os

from sklearn.model_selection import KFold
import numpy as np
from source.BiasLayer import BiasLayer
from source.DataSet import de_normalize


class MDACNN:
    def __init__(self, stack):
        self.model = self.mdacnn(stack.InputShape) # defined but NN model
        self.predictions = None
        self.hist = None

    def mdacnn(self, input_shape):

        '''define architecture of NN'''

        X_input = layers.Input(input_shape)

        #Stage 1

        X = layers.Conv2D(64, (3, 26), strides=1, name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)

        X = layers.BatchNormalization(axis=-1, name='bn_conv1')(X)
        X = layers.AveragePooling2D(pool_size=(5, 1), strides=3, padding="same")(X)

        X = layers.Flatten()(X)
        X = layers.Dropout(0.9)(X)

        #Stage 2.1
        X1 = layers.Dense(1)(X)

        #Stage 2.2
        X2 = layers.Dense(32, activation='tanh')(X) # #1 layer
        X2 = layers.Dense(32, activation='tanh')(X2) # #2 layer
        X2 = layers.Dropout(0.5)(X2)
        X2 = layers.Dense(32, activation='tanh')(X2)  # #2 layer
        X2 = layers.Dense(32, activation='tanh')(X2)  # #2 layer
        X2 = layers.Dropout(0.5)(X2)
        X2 = layers.Dense(32, activation='tanh')(X2)  # #2 layer
        X2 = layers.Dense(32, activation='tanh')(X2)  # #2 layer
        X2 = layers.Dropout(0.5)(X2)
        X2 = layers.Dense(32, activation='tanh')(X2)  # #2 layer
        X2 = layers.Dense(1)(X2) # #3 layer == linear activation.

        #Stage 3
        X = layers.Add()([X1, X2]) # add together
        X = layers.Dense(1)(X) # Output Layer

        model = Model(inputs=X_input, outputs=X, name='MDA_CNN')

        return model


    def launch_mdacnn(self, stack, train, val = None, fold_no = None, train__split = None):

        '''train + save NN'''

        stack.training = True  # needed to plot learning results on training data

        # hyperparameters for training
        self.model.compile(loss='mse', #mse
                          optimizer='adam',
                          metrics=['mse'])

        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=stack.checkpoint_path,
        #                                                 monitor='val_accuracy',
        #                                                 mode='max',
        #                                                 save_weights_only=True,
        #                                                 verbose=1)
        cp_callback = keras.callbacks.EarlyStopping(monitor='val_mse',  # Early Stopping
                                                    patience=stack.patience)

        if stack.modus == "Train+Test":


            # training
            #print("trainset.TrainDataset.shape:", train.feature.shape)
            #print("stack.InputShape:", stack.InputShape)
            self.hist = self.model.fit(train.feature, # Features
                              train.target, # force x component
                              epochs=stack.epochs, # Number of epochs
                              verbose=1,
                              batch_size=stack.BatchSize, #,
                              validation_data=(val.feature, val.target),
                              callbacks=[cp_callback])

            #self.model.save('mdacnn_trained.weights.h5')
            self.model.save(self.resource_path(stack.checkpoint_path))

        elif stack.modus == "Check Performance":


            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')

            train_feature, train_target = gather__Tensor(train__split, train)

            self.hist = self.model.fit(train_feature,#train.feature[train__split],
                                       train_target,#train.target[train__split],
                                       epochs=stack.epochs,
                                       verbose=1,
                                       batch_size=stack.BatchSize,
                                       validation_split= 0.1,
                                       callbacks=[cp_callback])


    def visualize_model_architecture(self):

        '''
        show the model architecture and the distribution of the parameters.
        Store an image of the model architecture in the "model.png" file
        '''

        print(self.model.summary())
        tf.keras.utils.plot_model(self.model, show_shapes=True)
        sys.exit()

    def resource_path(self, relative_path):

        '''
        return total path for a given relative path
        total path is "pyinstaller conform"
        '''

        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def load_model(self, stack):

        ''' load trained parameters out of .weights.h5 file '''

        #self.model.load_weights(self.resource_path(stack.checkpoint_path))
        self.model = load_model(self.resource_path(stack.checkpoint_path))


    def average_predictions(self, average_over_N_values):

        '''get list with predictions. Average each N elements together to one mean µ'''

        return list(np.mean(np.array(self.predictions).reshape(-1, average_over_N_values), axis=1))

    def predicts(self, features):

        ''' propagate test set through trained NN + obtains MDACNN values for input samples'''

        predictions = self.model.predict(features)  # prediction
        self.predictions = [k for i in predictions.tolist() for k in i]


    def evaluates(self, test):

        '''propagates test set through trained NN + obtains test loss and test accuracy'''

        self.model.compile(loss='mse',
                          optimizer='adam',
                          metrics=['mae'])

        test.target = tf.convert_to_tensor(test.target)
        test_loss, test_accuracy = self.model.evaluate(test.feature, test.target)
        return test_loss, test_accuracy

    def DefineParams(self, test):

        '''
        predefine all parameters needed to plot the ...
        prediction (MDACNN outputs)
        + the learning process (course of the error during the training process)
        '''

        # test.feature |--> datatable with 26 columns of order [LF,LF(LF), HF, LF(HF)]
        # test.target |--> ground truth value for yHF. HF-Function results. Result which should be achieved by MDACNN
        # test.feat_HF__target_LF |--> LF(HF) aka. most right column in data frame
        # self.predictions |--> Y_P for test.feature by MDACNN
        # all samples are sorted

        #y_hat = test.target
        #y_pred = self.predictions
        #accuracy = mean_squared_error(y_hat, y_pred)

        test.target = [x[0] for x in list(test.target.numpy())]  # Tensor |--> List
        # test.target__LF = test.target__LF[test.target__LF.columns[-1]].to_list() # DataFrame |--> List
        # test.feat_HF__target_LF = test.feat_HF__target_LF[test.feat_HF__target_LF.columns[-1]].to_list()
        output_values = [list(test.target), list(test.target__LF), self.predictions]  # list of lists
        output_values__flat = [k for i in output_values for k in i]  # flatten
        output_values__flat = [k for k in output_values__flat if
                               type(k) is not str]  # delete strings, keep float and int

        # print(output_values__flat)
        Y_min = min(np.array(output_values__flat))
        Y_max = max(np.array(output_values__flat))
        X_max = max(list(range(test.feature.shape[0])))  # Batch-Dim of Tensor |--> biggest Batch index
        X_min = min(list(range(test.feature.shape[0])))
        X = list(range(test.feature.shape[0]))

        return X, test, X_min, X_max, Y_min, Y_max


    #def PlotPrediction(self, X, test, X_min, X_max, Y_min, Y_max):
    def PlotPrediction(self, stack, test, str):
        '''
        plot the predictions made by the MDACNN
        The predictions are the output of the MDACNN for input features (which get forwarded through NN).
        '''

        plt.style.use("seaborn-v0_8")
        # print("style.available:", plt.style.available)

        #print(len(test.plot__TestOutput["X__MDACNN"]))
        #print(len(self.predictions))

        # E.g., originally 100 test tables. Due to cut+order (cut+padding) there are now 400 test tables.
        cut_ratio = int(len(self.predictions)/len(test.plot__TestOutput["X__MDACNN"]))

        self.predictions = np.array(self.predictions).reshape(-1, 1)
        self.predictions = de_normalize(self.predictions, test.scaler__HF_Pred)  # de-normalize the predicted data

        for plot_idx in range(cut_ratio):

            plt.plot(test.plot__TestOutput["X__LF"], test.plot__TestOutput["Y__LF"], color='b', marker='.', label='LF') # HF features + LF evaluation
            plt.plot(test.plot__TestOutput["X__HF"], test.plot__TestOutput["Y__HF"], color='r', marker='.', label='HF')
            plt.plot(test.plot__TestOutput["X__MDACNN"], self.predictions[plot_idx::cut_ratio], color='g', linestyle='--', marker='.', label='MDACNN')

            plt.title(str)
            plt.xlabel("Samples")
            plt.ylabel("Value")
            plt.legend(["LF", "HF", "Approximated HF"])

            plt.grid(True)
            #plt.xlim(X_min, X_max)  # show only in range of all X values
            #plt.ylim(Y_min, Y_max)  # show only range of all Y values

            #y_hat = np.array(test.target[plot_idx::cut_ratio])
            #print(type(y_hat))
            #sys.exit()
            y_hat = de_normalize(np.array(test.target[plot_idx::cut_ratio]).reshape(-1,1), test.scaler__HF_Pred)
            y_pred = self.predictions[plot_idx::cut_ratio]

            loss = mean_squared_error(y_hat, y_pred)

            #textstr = 'Accuracy=%.2f' % accuracy
            print(plot_idx,"... Loss (MSE):", loss)

            # plt.text(X_min - (np.sqrt(X_min ** 2 + X_max ** 2) / 8), Y_min + (np.sqrt(Y_min ** 2 + Y_max ** 2) / 2), textstr, fontsize=14)
            plt.subplots_adjust(left=0.25)
            plt.show()
            if (str == "MDACNN - Train Dataset") & (stack.training): break

    def plot__accuracy(self):

        '''
        Accuracy documents overfitting.
        Overfitting ... no generalization ... learning training samples instead of pattern
        '''

        plt.plot(self.hist.history['mse'], label="train")
        plt.plot(self.hist.history['val_mse'], label="val")
        plt.title("Accuracy")
        plt.legend()
        plt.show()

    def plot__loss(self):

        '''
        Loss documents underfitting.
        Underfitting ... not learning from training dataset. No learning samples and pattern.
        '''

        plt.plot(self.hist.history['loss'], label="train")
        plt.plot(self.hist.history['val_loss'], label="val")
        plt.title("Loss")
        plt.legend()
        plt.show()

    def PlotTrainingProgress(self):

        ''' plot the history of loss and accuracy during the training epochs'''

        self.plot__accuracy()
        self.plot__loss()

    def plot(self, stack, data, str):

        '''propagate test data through the trained model an plot results'''

        X, data, X_min, X_max, Y_min, Y_max= self.DefineParams(data)
        self.PlotPrediction(stack, data, str)

        if (self.hist is not None) & (str=="MDACNN - Test Dataset"): self.PlotTrainingProgress()

        test_loss, test_accuracy = self.evaluates(data)
        print(f" Training Loss (model.evaluate): {test_accuracy}")

    def analyse(self, stack, train, test):

        '''
        define data intervals (biggest, smallest sample value and biggest and smallest output value)
        + plot LF, HF and MDACNN output results
        '''

        self.plot(stack, test, "MDACNN - Test Dataset")
        if stack.training:
            self.predicts(train.feature)
            self.plot(stack, train, "MDACNN - Train Dataset")

        #X, test, X_min, X_max, Y_min, Y_max= self.DefineParams(test)

        #self.PlotPrediction( test)

        #if self.hist is not None: self.PlotTrainingProgress()

        #test_loss, test_accuracy = self.evaluates(test)


        #print(f" Training Loss (model.evaluate): {test_accuracy}")

def gather__Tensor(split, check):

    split = tf.convert_to_tensor(split)
    feature = tf.gather(check.feature, split, axis=0)
    target = tf.gather(check.target, split, axis=0)

    return feature, target

def cross_validation(stack, check):


    # Merge inputs and targets
    #inputs = np.concatenate((input_train, input_test), axis=0)
    #targets = np.concatenate((target_train, target_test), axis=0)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=stack.num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    for kfold, (train__split, test__split) in enumerate(kfold.split(check.feature, check.target)):
        # Define the model architecture
        MDACNN_Model = MDACNN(stack)  # define MDACNN

        MDACNN_Model.launch_mdacnn(stack = stack, train = check, fold_no = fold_no, train__split=train__split)


        test_feature, test_target = gather__Tensor(test__split, check)

        MDACNN_Model.predicts(test_feature)
        loss = mean_squared_error(MDACNN_Model.predictions, test_target)

        #scores = MDACNN_Model.model.evaluate(test_feature, test_target, verbose=0)
        print(f'Loss for fold {fold_no}: {loss}')
              #f'{MDACNN_Model.model.metrics_names[0]} of {scores[0]}; {MDACNN_Model.model.metrics_names[1]} of {scores[1] * 100}%')
        #acc_per_fold.append(scores[1])# * 100)
        loss_per_fold.append(loss)

        # Increase fold number
        fold_no = fold_no + 1

        MDACNN_Model.model.save(MDACNN_Model.resource_path(f'Weights__NN/wg_{kfold}.keras'))

    idx_max = loss_per_fold.index(min(loss_per_fold))
    file = f'Weights__NN/wg_{idx_max}.keras'

    #return np.mean(acc_per_fold), np.mean(loss_per_fold), file
    return np.mean(loss_per_fold), file

