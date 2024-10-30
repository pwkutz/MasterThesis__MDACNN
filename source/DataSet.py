import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import shap
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Dataset:
    def __init__(self, stack, amount_HFsamples,
                 feat, target__LF, target__HF):

        '''all input samples (locations) for training and test set'''

        self.RandomLocations = self.get_random_locations(feat.index, len(feat.index) - amount_HFsamples)

        self.plot__TestOutput = {"X__HF": list(range(len(list(target__HF)))),
                                 "Y__HF": target__HF,
                                 "X__LF": list(range(len(list(target__HF)))),
                                 "Y__LF": target__LF}

        '''Dataset'''

        (self.feat__LF, self.target__LF, # feature__LF + target__LF ... chosen with index for LF
         self.feat__HF, self.feat_HF__target_LF, # feature__HF + target__HF_LF ... chosen with index for HF
         self.target, self.scaler__HF_Pred) = self.HF_LF(feat, target__LF, target__HF) # target ... HF which must be predicted

        self.feature = self.DataTable(stack)  # 4-Rank Tensor



    def get_random_locations(self, figures, amount_samples):

        '''
        Randomly sample over a list of index values
        Return a sorted list of indeces
        '''

        return sorted(random.sample(list(figures), amount_samples))

    def HF_LF(self, feat, target__LF, target__HF):

        '''retrieve measurement values and evaluations for HF and LF'''

        # normalize data
        feat, target__LF, target__HF, scaler__HF = normalize_data(feat, target__LF, target__HF)
        feat, target__LF, target__HF = to_DataFrame(feat, target__LF, target__HF)

        # get Training Feature Part (most left part aka. yL)

        df__features__LF = feat.iloc[self.RandomLocations].reset_index(drop=True)

        # get Training Target Part (LF(yL))

        df__target__LF = pd.DataFrame()
        df__target__LF["LF"] = target__LF.iloc[self.RandomLocations].reset_index(drop=True)
        #print(df__train_target__LF)

        # get Training Target Part middle (middle right yHF)

        HF__index = sorted(set(list(range(len(feat)))) - set(self.RandomLocations))
        self.plot__TestOutput["X__MDACNN"] = HF__index #save for plotting issues


        df__features__HF = feat.iloc[list(HF__index)].reset_index(drop=True)

        # get right LF(yHF)

        df__target__HF_LF = pd.DataFrame()
        df__target__HF_LF["LF(HF)"] = target__LF.iloc[list(HF__index)].reset_index(drop=True)

        # Ground Truth each HF location (ground truth HF(yHF))

        df__target__HF = pd.DataFrame()
        df__target__HF["HF"] = target__HF.iloc[list(HF__index)].reset_index(drop=True)

        return df__features__LF, df__target__LF, df__features__HF, df__target__HF_LF, df__target__HF, scaler__HF



    def DataTable(self, stack):

        ''' compute final DataTables organized in a Tensors.
        DataTables are the final input into the MDACNN '''

        list__data_tables = []

        self.feat__HF.columns = (self.feat__HF.columns +
                                               max(self.feat__HF.columns))  # no doubling columns

        # print(trainset.df__train_features__HF)

        for index in range(len(self.feat__HF)):
            feat__HF = prepair_HF(self.feat__HF, self.feat__LF, index)
            target__HF_LF = prepair_HF(self.feat_HF__target_LF, self.target__LF, index)

            df = pd.DataFrame()
            df = pd.concat([self.feat__LF, self.target__LF,
                            feat__HF, target__HF_LF], axis=1)  # put build input table
            list__data_tables.append(df) # list of tables

        self.feature = list__data_tables
        self.CutAndOrderData(stack)

        return self.feature


    def CutAndOrderData(self, stack):

        '''generate equally sized DataFrames. Pad if necessary'''

        list__data_tables = []
        list__targets = []
        list__feats_HF__targets_LF = []

        list_g = []
        for idx, test in enumerate(self.feature):
            list__features, list__target, list__feat_HF__target_LF = self.cut(idx, stack, test)

            list__data_tables.append(list__features)
            list__targets.append(list__target)
            list__feats_HF__targets_LF.append(list__feat_HF__target_LF)


        list__data_tables = [x for xs in list__data_tables for x in xs]
        list__targets = [x for xs in list__targets for x in xs]
        list__feats_HF__targets_LF = [x for xs in list__feats_HF__targets_LF for x in xs]

        self.feature = list__data_tables
        self.target = list__targets
        self.feat_HF__target_LF = list__feats_HF__targets_LF



    def cut(self, idx, stack, test):

        '''cut original DataFrame into smaller, equally sized DataFrames'''

        n = stack.amount_LFsamples__Training
        list__data_tables__cut = []

        for g, df in test.groupby(np.arange(len(test)) // n):  # cut DataFrame into DataFrames with less rows
            df = df.reset_index(drop=True)
            if g == max(np.arange(len(test)) // n):  # last dataframe: check whether it has equal sized as other sub-df
                df = pad(df, stack)


            list__data_tables__cut.append(df)

        list__target__cut = (g+1) * list(self.target.iloc[idx]) # Split 1 table into 4 |--> 4 x ground truth
        list__feat_HF__target_LF__cut = (g+1)*list(self.feat_HF__target_LF.iloc[idx])

        return list__data_tables__cut, list__target__cut, list__feat_HF__target_LF__cut

    def ToTensor(self):
        self.feature = ListOfDataFrames_To_Tensor(self.feature)  # 4-Rank Tensor
        self.target = ListOfDataFrames_To_Tensor(self.target)  # 4-Rank Tensor


def moving_average(window_size, Pandas_Series):

    '''apply a moving average to a given series of values. Goal: smoothing + denoising the series'''

    Pandas_Series = Pandas_Series.rolling(window=window_size, center=True).mean()
    return Pandas_Series.dropna()

def apply__moving_average(window_size, feat, LF, HF):

    '''
    apply moving average to the train, val and test dataset
    COMMENTS: decomment to plot effect of Moving Average
    '''

    #a = LF
    #b = HF

    LF = moving_average(window_size, LF)
    HF = moving_average(window_size, HF)
    common_idx = feat.index.intersection(LF.index)
    feat = feat.loc[common_idx]

    LF = LF.reset_index(drop=True)
    HF = HF.reset_index(drop=True)
    feat = feat.reset_index(drop=True)

    #plt.plot(list(range(len(a))), a, color='c')
    #plt.plot(list(range(len(b))), b, color='c')
    #plt.plot(list(range(len(HF))), HF, color='b')
    #plt.plot(list(range(len(LF))), LF, color='r')
    #plt.show()

    return feat, LF, HF


def fetch__target(stack):

    '''
    define ground truth value
    MDACNN gets trained to learn the course of this parameter.
    '''

    return stack.YYaw # XTraction, ZNormal, YYaw

def fetch__TrainSource():

    '''get Train data from the source file'''

    train = pd.read_csv('/Users/philippkutz/Desktop/Weiterbildung/TUM/RCI_SS_24/Masterarbeit/OfficialTrainingData/'
                          'mufintroll_tabular_data/HF_w_LF_tabular_data.csv', header= None)
    return train

def fetch__TestSource():

    '''get Test data from the source file'''

    test = pd.read_csv('/Users/philippkutz/Desktop/Weiterbildung/TUM/RCI_SS_24/Masterarbeit/OfficialTrainingData/'
                          'mufintroll_tabular_data/HF_w_LF_test_tabular_data.csv', header= None)
    return test


def get_TrainData(train, target_feature):

    '''retrieve 12-dim data measurements + their LF and HF target value (ground truth)'''

    train_features = train.iloc[:, :12]  # retrieve training data (input for forward propagation)
    train_outputs = train.iloc[:, 12:]  # LF + HF force and torque predictions
    LF_to_HF = int(len(train_outputs.columns) / 2)
    train_target__LF = train_outputs.iloc[:, target_feature]  # x directional traction force
    train_target__HF = train_outputs.iloc[:, target_feature + LF_to_HF]

    return train_features, train_target__LF, train_target__HF

def get_TestData(test, target_feature, index):

    if index is not None:
        run_indx = test.iloc[:,-1]== index
        features = test.iloc[:,:12][run_indx].reset_index(drop=True)
        outputs = test.iloc[:,12:][run_indx].reset_index(drop=True)
    else:
        features = test.iloc[:,:12].reset_index(drop=True)
        outputs = test.iloc[:,12:].reset_index(drop=True)

    LF_to_HF = int(len(outputs.columns) / 2)
    target__LF = outputs.iloc[:, target_feature] # 0...XTraction, 2...ZNormal, 4...YYaw
    target__HF = outputs.iloc[:,target_feature+LF_to_HF]

    return features, target__LF, target__HF



def retrieve_BasicData(stack):

    '''
    Pre-process raw input data.
    Make it usable for training, validating and testing
    '''

    target_feature = fetch__target(stack) # XTraction, ZNormal, YYaw

    train = fetch__TrainSource()

    train_features, train_target__LF, train_target__HF = get_TrainData(train, target_feature)
    train_features, train_target__LF, train_target__HF = (apply__moving_average(stack.moving_average__window["Train"],
                                                                                train_features, train_target__LF,
                                                                                train_target__HF)) # Moving Average

    test = fetch__TestSource()

    val_features, val_target__LF, val_target__HF = get_TestData(test, target_feature, 4558)
    val_features, val_target__LF, val_target__HF = (
        apply__moving_average(stack.moving_average__window["Val"], val_features, val_target__LF, val_target__HF)) # Moving Average

    test_features, test_target__LF, test_target__HF = get_TestData(test, target_feature, 4343)
    test_features, test_target__LF, test_target__HF = (
        apply__moving_average(stack.moving_average__window["Test"], test_features, test_target__LF, test_target__HF)) # Moving Average

    #return (train_features, train_target__LF, train_target__HF,
    #        test_features, test_target__LF, test_target__HF,
    #        val_features, val_target__LF, val_target__HF)

    return (train_features, train_target__LF, train_target__HF,
            val_features, val_target__LF, val_target__HF,
            test_features, test_target__LF, test_target__HF)

def retrieve_CommonDataset(stack):

    '''prepair Train- and Test-Dataset for K-Fold Cross Validation'''

    target_feature = fetch__target(stack) # XTraction, ZNormal, YYaw

    train = fetch__TrainSource() # get Train-Dataset
    train_features, train_target__LF, train_target__HF = get_TrainData(train, target_feature)

    test = fetch__TestSource() # get whole Test-Dataset
    test_features, test_target__LF, test_target__HF = get_TestData(test, target_feature, None)

    features = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)  # retrieve common Dataset
    target__LF = pd.concat([train_target__LF, test_target__LF], axis=0).reset_index(drop=True)
    target__HF = pd.concat([train_target__HF, test_target__HF], axis=0).reset_index(drop=True)

    features, target__LF, target__HF = (apply__moving_average(stack.moving_average__window["Train"], # Moving Average
                                                              features, target__LF,target__HF))

    return features, target__LF, target__HF

def check(stack, test_features):
    if (len(test_features) - stack.amount_HFsamples__Testing) < stack.amount_LFsamples__Testing:
        print("Test + Validation Data ... active rows (no padding):", (len(test_features) - stack.amount_HFsamples__Testing))
        print("Rows with padding:", stack.amount_LFsamples__Training)
        print("Recommendation: lower", stack.amount_LFsamples__Training,"to", (len(test_features) - stack.amount_HFsamples__Testing),"rows")
        print("Change parameters 'stack.amount_LFsamples__Training' ")
        #print("Maximum amount of allowed rows aka. samples for input data:", len(test_features))
        #print("Parameter to change: 'self.amount_LFsamples__Training'")
        sys.exit("ERROR: input data shape to big. To many rows. Lower amount rows.")

def prepair_HF(dataset__HF, dataset__LF, index):
    '''
    get dataframes consisting out of the same row
    HF training data part of the input data.
    '''

    len__df = len(dataset__HF)  # save length

    new_row = dataset__HF.iloc[index].copy()  # save row + information regarding row location in dataframe
    dataset__HF = dataset__HF._append([new_row] * len(dataset__LF), ignore_index=True)  # add row N times
    dataset__HF = dataset__HF.iloc[len__df:].reset_index(drop=True)  # delete all proir columns

    return dataset__HF

def pad(df, stack):

    '''add  0 rows to a Pandas DataFrame'''

    out = pd.concat([df, pd.DataFrame(0, columns=df.columns,
                                      index=np.arange(max(df.index) + 1, stack.amount_LFsamples__Training))]
                    ).reset_index(drop=True)

    return out

def get_Tensor(ListOfDataFrames):

    '''shape out of List of DataFrames a single 4-Rank Tensor'''

    TableList_Array = np.array(ListOfDataFrames)  # List --> Append everything. Array --> Transpose. Tensor --> stack and input into NN
    TableList_T = []

    for table in range(len(ListOfDataFrames)):
        arrayT = TableList_Array[table]#.T  # Transpose. Each table individually
        TableList_T.append(arrayT)

    return tf.stack(TableList_T)  # Training Table == Input Tensor into NN

def reshape_Tensor(Tensor):

    '''
    :param Tensor: 4-Rank Tensor with a 3D shape (Channel=1 ==> Channel dim gets cut)
    :return: Tensor: 4-Rank Tensor with a 3D shape [(B x H x W x C) where (B x H x W x 1)]
    '''

    new_shape = []
    new_shape.append(list(Tensor.shape))
    new_shape.append([1])
    new_shape = [x for sublist in new_shape for x in sublist]
    Tensor = tf.reshape(Tensor, new_shape)

    return Tensor


def ListOfDataFrames_To_Tensor(ListOfDataFrames):

    '''
    Retrieve out of List of DataFrames a 4-Rank Tensor.
    Reshape 4-Rank Tensor. Get guaranteed 4D dimensional shape.
    '''
    Tensor = get_Tensor(ListOfDataFrames)
    Tensor = reshape_Tensor(Tensor)

    return Tensor
    #TableList_Array = np.array(ListOfDataFrames)  # List --> Append everything. Array --> Transpose. Tensor --> stack and input into NN
    #TableList_T = []

    #for table in range(len(ListOfDataFrames)):
    #    arrayT = TableList_Array[table]#.T  # Transpose. Each table individually
    #    TableList_T.append(arrayT)

    #return tf.stack(TableList_T)  # Training Table == Input Tensor into NN

def normalize(df):

    '''
    Normalize Pandas object column-wise
    Each column gets normalized separately
    '''

    #normalized_df = (df - df.min()) / (df.max() - df.min())
    #return normalized_df.fillna(0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    return scaled, scaler

def de_normalize(df, scaler):
    unscaled = scaler.inverse_transform(df)
    return unscaled

def normalize_data(feat, target__LF, target__HF):

    feat, _ = normalize(feat)
    target__LF, _ = normalize(pd.DataFrame(target__LF))
    target__HF, scaler__HF = normalize(pd.DataFrame(target__HF))

    return feat,  target__LF, target__HF, scaler__HF

def turn_to_DataFrame(array):

    ''' return a default Pandas DataFrame without specified column names '''

    return pd.DataFrame(array)

def to_DataFrame(feat, target__LF, target__HF):

    '''
    After normalization are all features, and targets Numpy arrays.
    For further processing those need to be Pandas DataFrame. Therefore, transform them here.
    '''

    feat = turn_to_DataFrame(feat)
    target__LF = turn_to_DataFrame(target__LF)
    target__HF = turn_to_DataFrame(target__HF)

    return feat, target__LF, target__HF

def FeatureImportance__Shap(MDACNN_Model, train, test):

    column__names = list(range(train.feature.shape[-2]))

    DE = shap.DeepExplainer(MDACNN_Model.model, train.feature.numpy())  # X_train is 3d numpy.ndarray
    shap_values = DE.shap_values(test.feature.numpy(), check_additivity=False)  # X_validate is 3d numpy.ndarray

    ShapValue = shap_values.reshape(shap_values.shape[0:-1])
    ShapValue = ShapValue.reshape(ShapValue.shape[0:-1])

    results__mean = []
    results__var = []
    for batch in ShapValue:
        batch__mean = batch.mean(axis=0)
        batch__var = batch.var(axis=0)
        results__mean.append(batch__mean)
        results__var.append(batch__var)

    results__mean = np.array(results__mean).mean(axis=0)
    results__var = np.array(results__var).mean(axis=0)
    mean = dict(zip(column__names[:12], results__mean[:12]))
    var = dict(zip(column__names[:12], results__var[:12]))

    return mean, var

def FeatureImportance__results_and_exit(mean, var):

    print(mean)
    print(var)
    sys.exit()
