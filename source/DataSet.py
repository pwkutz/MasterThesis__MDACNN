import sys

import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class Dataset:
    def __init__(self, stack):

        '''TrainDataset'''
        self.RandomLocations = self.get_random_locations(stack, stack.upper_boundary, stack.lower_boundary, stack.amount_LFsamples + stack.amount_HFsamples) #Training
        self.LFdataset = self.LFfunction(self.RandomLocations[:stack.amount_LFsamples], stack) # QL(yLi) ... LF function, LF locations. List.
        self.LFdatasetHFsamples = self.LFfunction(self.RandomLocations[-stack.amount_HFsamples:], stack) # QL(yHi) ... LF function, HF locations. List.
        self.HFdataset = self.HFfunction(self.RandomLocations[-stack.amount_HFsamples:], stack) # QH(yHi) + ground truth predictions. 1-Rank Tensor.
        if not stack.experimentG: # normalize data only for Experiments A - F
            (self.RandomLocations, self.RandomLocations__scaler,
             self.LFdataset, self.LFdataset__scaler,
             self.LFdatasetHFsamples, self.LFdatasetHFsamples__scaler,
             self.HFdataset, self.HFdataset__scaler) = self.normalization(self.RandomLocations, self.LFdataset,
                                                                          self.LFdatasetHFsamples, self.HFdataset)
        self.TrainDataset = self.DataTable(stack, self.RandomLocations, self.LFdataset, self.LFdatasetHFsamples) # 4-Rank Tensor


        '''TestDataset'''
        self.RandomLocationsTest = self.get_random_locations(stack, stack.upper_boundary, stack.lower_boundary, stack.amount_LFsamples + stack.amount_HFsamples)  # Test
        self.LFdatasetTest = self.LFfunction(self.RandomLocationsTest[:stack.amount_LFsamples], stack)
        self.LFdatasetHFsamplesTest = self.LFfunction(self.RandomLocationsTest[-stack.amount_HFsamples:], stack)
        self.HFdatasetTest = self.HFfunction(self.RandomLocationsTest[-stack.amount_HFsamples:], stack)
        if not stack.experimentG: # normalize data only for Experiments A - F
            (self.RandomLocationsTest, self.RandomLocationsTest__scaler,
             self.LFdatasetTest, self.LFdatasetTest__scaler,
             self.LFdatasetHFsamplesTest, self.LFdatasetHFsamplesTest__scaler,
             self.HFdatasetTest, self.HFdatasetTest__scaler) = self.normalization(self.RandomLocationsTest, self.LFdatasetTest,
                                                                                  self.LFdatasetHFsamplesTest, self.HFdatasetTest)

        self.TestDataset = self.DataTable(stack, self.RandomLocationsTest, self.LFdatasetTest, self.LFdatasetHFsamplesTest)

    def normalization(self, RandomLocations, LF, LFHF, HF):

        '''
        normalize features, LF, and HF values
        enable a more effective way of training by re-scale all features to a common scale
        '''

        loc, scaler__loc = normalize(pd.DataFrame(RandomLocations))
        LF, scaler__LF = normalize(pd.DataFrame(LF))
        LFHF, scaler__LFHF = normalize(pd.DataFrame(LFHF))
        HF, scaler__HF = normalize(pd.DataFrame(HF.numpy()))

        HF = tf.convert_to_tensor(HF) # Transform HF function values back to TF-Tensor after normalization

        return (loc, scaler__loc, #feat
                LF, scaler__LF, # LF function values
                LFHF, scaler__LFHF, # HF feat with LF function values
                HF, scaler__HF) # HF function values (ground truth)




    def get_random_locations(self, stack, upper_boundary, lower_boundary, amount_samples):
        if not stack.experimentG:
            return random.sample(list(np.linspace(lower_boundary,upper_boundary,10000)), amount_samples)
        else:
            list_OfSamples=[]
            list_h = []
            for sample in range(amount_samples):
                list_h = random.sample(list(np.linspace(lower_boundary, upper_boundary, 10000)), 50)
                list_OfSamples.append(list_h)

            return list_OfSamples

    def retrieve__LFValue_G(self, dim_50, stack):

        '''compute the LF function values for Experiment G (50-dimensional feature samples)'''

        sumG = compute_sum__LF(dim_50)
        res = 0.8 * self.HFfunction(dim_50, stack) - sumG - 50
        return res

    def retrieve__HFValue_G(self, dim_50):

        '''compute the HF function values for Experiment G (50-dimensional feature samples)'''

        sumG = compute_sum__HF(dim_50)
        res  = (dim_50[0] - 1)**2 + sumG
        return res


    def LFfunction(self, RandomLocations, stack):
        '''each experiment (A,B,C, D) possesses its own LF function'''
        if stack.experimentA or stack.experimentC:

            # Experiment A: Continuous functions with linear relationship
            # Experiment C: Continuous functions with nonlinear relationship

            return [(0.5*((6*y -2)**2)*np.sin(12*y - 4) + 10*(y - 0.5) - 5) for y in RandomLocations]

        elif stack.experimentB:

            # Experiment B: Discontinuous functions with linear relationship

            values = []
            for y in RandomLocations:
                if 0 <= y <= 0.5: values.append(0.5*((6*y - 2)**2)*np.sin(12*y - 4) + 10*(y - 0.5))
                else: values.append(3 + 0.5*((6*y - 2)**2)*np.sin(12*y - 4) + 10*(y - 0.5))
            return values

        elif stack.experimentD or stack.experimentE:

            # Experiment D: Continuous oscillation functions with nonlinear relationship
            # Experiment E: Phase-shifted oscillations

            return [(np.sin(8*np.pi*y)) for y in RandomLocations]

        elif stack.experimentF:

            # Experiment F: Different periodicities

            return [(np.sin(6*np.sqrt(2)*np.pi*y)) for y in RandomLocations]

        elif stack.experimentG:

            return [self.retrieve__LFValue_G(y, stack) for y in RandomLocations]

        else:
            print("Define experiment. Use instance 'stack' of class Parameters")
            sys.exit()


    def HFfunction(self, RandomLocations, stack):
        '''each experiment (A,B,C, D) possesses its own HF function'''
        if stack.experimentA:

            # Experiment A: Continuous functions with linear relationship
            return tf.convert_to_tensor(np.array([(((6*y - 2)**2)*np.sin(12*y - 4)) for y in RandomLocations]))

        elif stack.experimentB:

            # Experiment B: Discontinuous functions with linear relationship

            values = []
            for y in RandomLocations:
                if 0 <= y <= 0.5:
                    values.append(2 * self.LFfunction([y], stack)[0] - 20*y + 20)
                else:
                    values.append(4 + 2 * self.LFfunction([y], stack)[0] - 20*y + 20)
            return tf.convert_to_tensor(np.array(values))

        elif stack.experimentC:

            # Experiment C: Continuous functions with nonlinear relationship

            return tf.convert_to_tensor(np.array([((6*y - 2)**2*np.sin(12*y - 4) - 10*(y - 1)**2) for y in RandomLocations]))


        elif stack.experimentD:

            # Experiment D: Continuous oscillation functions with nonlinear relationship

            return tf.convert_to_tensor(np.array([((y - np.sqrt(2)) * (self.LFfunction([y], stack)[0])**2) for y in RandomLocations]))

        elif stack.experimentE:

            # Experiment E: Phase-shifted oscillations

            return tf.convert_to_tensor(np.array([(y**2 + (self.LFfunction([y + 10/np.pi], stack)[0])**2) for y in RandomLocations]))

        elif stack.experimentF:

            # Experiment F: Different periodicities

            return tf.convert_to_tensor(np.array([(np.sin(8*np.pi*y + np.pi/10)) for y in RandomLocations]))

        elif stack.experimentG:

            if type(RandomLocations[0]) is not list:
                return self.retrieve__HFValue_G(RandomLocations) # Needed to compute LF function value for G
            else:
                return tf.convert_to_tensor(np.array([self.retrieve__HFValue_G(y) for y in RandomLocations])) # needed to compute actual HF function value of G

        else:
            print("Define experiment. Use instance 'stack' of class Parameters")
            sys.exit()


    def DataTable(self, stack, RandomLocations, LFdataset, LFdatasetHFsamples, ):

        LFlist_Location = RandomLocations[:stack.amount_LFsamples] # yLi
        LFlist_Value = LFdataset # QL(yLi)

        HFTableLocation = []
        HFTableValue = []
        listH = []

        for HFlocation in RandomLocations[-stack.amount_HFsamples:]:  # yHi
            listH = [HFlocation] * stack.amount_LFsamples  # helper list listH. List contains same value n x times
            HFTableLocation.append(listH)

        for HFvalue in LFdatasetHFsamples:  # QL(yHi)
            listH = [HFvalue] * stack.amount_LFsamples
            HFTableValue.append(listH)

        import pandas as pd

        TableList = []
        if stack.experimentG: LFlist_Location = np.array(LFlist_Location).T.tolist()
        for number in range(stack.amount_HFsamples):
            if stack.experimentG:

                HFTableLocation[number] = np.array(HFTableLocation[number]).T.tolist()
                columns = differentiat_and_order(LFlist_Location, LFlist_Value, HFTableLocation[number], HFTableValue[number])
                TableList.append(columns)
            else:

                TableList.append([LFlist_Location, LFlist_Value, HFTableLocation[number], HFTableValue[number]])  # Put [yLi, QL(yLi), yHi, QL(yHi)] together. Need to Transpose.




        TableList_Array = np.array(TableList)  # List --> Append everything. Array --> Transpose. Tensor --> stack and input into NN

        TableList_T = []

        for table in range(len(TableList)):
            arrayT = TableList_Array[table].T  # Transpose. Each table individually
            arrayT = arrayT.reshape(stack.InputShape)
            TableList_T.append(arrayT)

        return tf.stack(TableList_T)  # Training Table == Input Tensor into NN



def compute_sum__LF(dim_50):

    '''
    compute sum which is needed in the computation of the LF and HF function value in Experiment G
    :param dim_50: 50-dimensional feature sample
    :return:
    '''

    list__product = []
    for index in range(len(dim_50)):
        if index is not len(dim_50) - 1: list__product.append(0.4 * dim_50[index] * dim_50[index + 1])
    sum_G = sum(list__product)

    return sum_G

def compute_sum__HF(dim_50):

    '''
    compute sum which is needed in the computation of the LF function value in Experiment G
    :param dim_50: 50-dimensional feature sample
    :return:
    '''

    list__product = []
    for index in range(len(dim_50)):
        if index != 0: list__product.append((2* dim_50[index]**2 - dim_50[index - 1])**2)
    sum_G = sum(list__product)

    return sum_G

def check__dimensions(stack, LFlist_Location):

    shape__original = np.array(LFlist_Location).shape
    if shape__original is not (stack.experimentG__dimension, stack.amount_LFsamples):
        return

def differentiat_and_order(LFlist_Location, LFlist_Value, HFTableLocation, HFTableValue):

    common_list = [LFlist_Location, LFlist_Value, HFTableLocation, HFTableValue]
    new_list = []

    def for_loop(sub_list):
        for sub_sub in sub_list:
            new_list.append(sub_sub)
    [for_loop(sub_list) if type(sub_list[0]) is list else new_list.append(sub_list) for sub_list in common_list]

    return new_list

def normalize(df):

    '''
    Normalize Pandas object column-wise
    Each column gets normalized separately
    '''

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    return scaled, scaler



