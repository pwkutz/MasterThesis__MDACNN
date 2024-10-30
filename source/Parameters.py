class Parameters:

    def __init__(self):

        self.modus = "Train+Test" # "Check Performance" (Cross Validation) | "Train + Test"
        #self.modus = "Check Performance"

        self.num_folds = 10 # K-Fold Cross-Validation
        self.moving_average__window = {"Train": 50, "Val": 20, "Test": 20} # Smooth train-, val- and test dataset
        self.patience = 10 # Early Stopping
        self.training = False # Propagate Train-Dataset through trained model.

        ''' Experiment related parameters. Default values: see Paper'Multi-fidelity Data Aggregation using Convolutional Neural Networks', Table 2. '''
        self.amount_LFsamples__Training = 195 # amount rows Training/Val/Test Tables
        self.amount_HFsamples__Training = 1000#1000  # amount Training Tables
        self.amount_LFsamples__Validation = self.amount_LFsamples__Training # amount rows Training/Val/Test Tables
        self.amount_HFsamples__Validation = 300 # amount Val Tables
        self.amount_LFsamples__Testing = self.amount_LFsamples__Training # amount rows Training/Val/Test Tables
        self.amount_HFsamples__Testing = 100  # amount Test Tables

        self.epochs = 75
        self.BatchSize = 32

        self.InputShape = (self.amount_LFsamples__Training, 26, 1) # Shape Input Layer == Shape of the input for a single propagation.
        self.checkpoint_path = "MDACNN_model.keras"  # store trained weights

        '''Indeces for parameters of interest. These parameters get trained by the NN'''
        self.XTraction = 0
        self.ZNormal = 2
        self.YYaw = 4


