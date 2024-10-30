import sys

from source.Parameters import Parameters
from source.MDACNN import MDACNN
from source.DataSet import Dataset

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    stack = Parameters()  # hyperparameters
    trainset = Dataset(stack)  # get locations x and LF + HF function values

    MDACNN_Model = MDACNN(stack)  # define MDACNN

    '''before training'''

    MDACNN_Model.launch_mdacnn(stack, trainset) # train and save MDACNN

    ''' after training'''

    MDACNN_Model.load_model(stack)
    MDACNN_Model.predicts(trainset.TestDataset) # propagate testset
    MDACNN_Model.analyse(stack, trainset) # accuracy + plotting

