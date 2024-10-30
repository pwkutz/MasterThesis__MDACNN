
from source.Modes import Train_And_Test, Performance_Check
from source.Parameters import Parameters


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    stack = Parameters()  # hyperparameters


    if stack.modus == "Train+Test": Train_And_Test(stack)# model.fit + model.predict
    elif stack.modus == "Check Performance": Performance_Check(stack) # K-Fold Cross Validation







