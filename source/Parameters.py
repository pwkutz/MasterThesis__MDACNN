class Parameters:
    def __init__(self):


        ''' Experiment related parameters. Default values: see Paper'Multi-fidelity Data Aggregation using Convolutional Neural Networks', Table 2. '''
        self.amount_LFsamples= 21#1000 # amount rows Training/Test Tables
        self.amount_HFsamples= 4 #amount Training/Test Tables
        self.epochs = 5000#1500
        self.BatchSize = 4
        self.checkpoint_path = "mdacnn_trained.weights.h5"  # store trained weights
        self.training = False # check whether the model got trained or not. Important for plotting learning results based on Training Data


        '''1:True. 3:False. 
        One experiment needs to be defined True, all others experiments need to be defined as False.'''
        # use different Low Fidelity and High Fidelity Functions to compute the values QL(y) and QH(y).
        self.experimentA = True#True#True
        self.experimentB = False#True
        self.experimentC = False#True
        self.experimentD = False#True#True
        self.experimentE = False#True
        self.experimentF = False#True#True
        self.experimentG = False#True

        if not self.experimentG:

            self.lower_boundary = 0  # location: 0 <= x <= 1
            self.upper_boundary = 1

            self.InputShape = (self.amount_LFsamples, 4, 1)  # Shape Input Layer == Shape of the input for a single propagation.
            self.KernelHeight = 3
            self.KernelWidth = 4
        else:
            self.lower_boundary = -3  # location: -3 <= x <= 3
            self.upper_boundary = 3

            self.InputShape = (self.amount_LFsamples, 102, 1)  # Shape Input Layer == Shape of the input for a single propagation.
            self.KernelHeight = 3
            self.KernelWidth = 102 # 50 + 1 + 50 + 1



