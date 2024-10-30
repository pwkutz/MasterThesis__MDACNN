import sys
import tensorflow as tf

from source.Parameters import Parameters
from source.MDACNN import MDACNN, cross_validation
from source.DataSet import (Dataset, retrieve_BasicData, retrieve_CommonDataset, FeatureImportance__Shap,
                            FeatureImportance__results_and_exit)


def Train_And_Test(stack):

    '''
    First Mode: First train Model using Train and Validation Data, then test trained Model using Test-Data

    :param stack:
    :return: None
    '''

    (train_features, train_target__LF, train_target__HF,
     val_features, val_target__LF, val_target__HF,
     test_features, test_target__LF, test_target__HF) = retrieve_BasicData(stack)  # roughly pre-processed input data

    train = Dataset(stack, stack.amount_HFsamples__Training,
                    train_features, train_target__LF, train_target__HF)  # train data
    val = Dataset(stack, stack.amount_HFsamples__Validation,
                  val_features, val_target__LF, val_target__HF)  # val data
    test = Dataset(stack, stack.amount_HFsamples__Testing,
                   test_features, test_target__LF, test_target__HF)  # test data

    train.ToTensor() # 4-Rank Tensor
    val.ToTensor()  # 4-Rank Tensor
    test.ToTensor()  # 4-Rank Tensor

    MDACNN_Model = MDACNN(stack)  # define MDACNN
    MDACNN_Model.visualize_model_architecture()

    #stack.BatchSize = train.feature.shape[0] # Batch-Gradient Descent

    '''training'''

    MDACNN_Model.launch_mdacnn(stack=stack, train=train, val=val)  # train and save MDACNN

    ''' feature importance ''' # de-comment "'''training'''" and "''' feature importance '''"

    # load your data here, e.g. X and y
    # create and fit your model here

    #mean, var  = FeatureImportance__Shap(MDACNN_Model, train, test)
    #FeatureImportance__results_and_exit(mean, var)


    ''' performance check '''

    MDACNN_Model.load_model(stack)
    MDACNN_Model.predicts(test.feature) # propagate testset
    MDACNN_Model.analyse(stack, train, test) # accuracy + plotting

def Performance_Check(stack):

    '''
    Second Mode: take Model arch, Train+Val+Test Dataset and apply K-Fold Cross Validation on it.
    Goal evaluate the model by its average Test-accuracy after training with differently split Train- and Test-Data
    :param stack:
    :return: None
    '''


    features, target__LF, target__HF = retrieve_CommonDataset(stack)

    amount_HFSamples = (stack.amount_HFsamples__Training
                        + stack.amount_HFsamples__Validation
                        + stack.amount_HFsamples__Testing)

    check = Dataset(stack, amount_HFSamples,
                    features, target__LF, target__HF)  # train data

    check.ToTensor()  # 4-Rank Tensor

    #stack.BatchSize = check.feature.shape[0]
    '''check performance'''
    average_MSE__loss, file__maxAccuracy = cross_validation(stack, check)

    print("Average (Test)-Loss:", average_MSE__loss)
    print("Model with optimal (Test-)Accuracy:", file__maxAccuracy)