import sys
import subprocess
import os
import shutil
import pandas as pd
from dataset_details import *

def accuracy_value(predicted_value):
    """
    compute the accuracy from predicted value and acutal value
    :param predicted_value: predicted value dataframe that has first column as predicted value and second column corresponding actual class label
    :return: int accuracy
    """
    positive = predicted_value[predicted_value["predictedValue"] == 1]
    negative = predicted_value[predicted_value["predictedValue"] == 0]
    True_positive = len(positive[(positive["predictedValue"] == positive["test_label"])])
    True_negative = len(negative[(negative["predictedValue"] == negative["test_label"])])
    False_positive = len(positive[(positive["predictedValue"] != positive["test_label"])])
    False_negative = len(negative[(negative["predictedValue"] != negative["test_label"])])
    accuracy = (True_positive + True_negative) / (len(predicted_value))
    print("Final Bagging Confusion matrix....")
    print("                Predicted:Negative(0)   Predicted: Positive(1)")
    print("Actual: Negative           {}            {}".format(True_negative, False_positive))
    print("Actual: Positive           {}            {}".format(False_negative, True_positive))
    print("Accuracy of the Bagging is: {}".format(accuracy))
    print("Misclassification error of the bagging is: {}".format(1 - accuracy))
    return accuracy

def bagging_tree(maxdepth,bags,datapath):
    """
    Generate ensemble bagging of given depth and bags
    :param maxdepth: depth of the tree
    :param bags: total required bags
    :param datapath: Localtion of the input data
    :return:
    """

    #initialize empty list to store weights
    for i in range(int(bags)):
       if(i==0):
          currDir = os.path.dirname(os.path.realpath('__file__'))
          if not os.path.exists('temp'):
             os.makedirs('temp')
          else:
             shutil.rmtree('temp')
             os.makedirs('temp')

       #call another script that will generate trees of given depth
       theproc = subprocess.Popen([sys.executable, "decision_tree_bagging.py",str(maxdepth),str(datapath)])
       theproc.communicate()

    #write all the output predicted value files and combine them into single dataframe
    train, test, depth = dataset_read_bagging(maxdepth,datapath)
    os.chdir(os.getcwd() + "/temp/")
    filelist = os.listdir(os.getcwd())
    df_list = [pd.read_csv((os.getcwd() + "/" + file)) for file in filelist]
    combined_df = pd.concat(df_list, axis=1)
    combined_df = combined_df.mode(axis=1)

    #combine datafram
    combined_df = pd.concat([combined_df,test.loc[:,'class']],axis=1)
    combined_df.columns = ['predictedValue','test_label']
    combined_df.to_csv('combi.csv',index=False)
    acc = accuracy_value(combined_df)
    # remove created directory
    os.chdir("..")
    shutil.rmtree('temp')
    shutil.rmtree('__pycache__')

