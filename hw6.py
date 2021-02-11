
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
import keras as k
import numpy as np
import pandas as pd
import sklearn.preprocessing as skp
import time

def addOutputLayer(model,numClasses):
    model.add(Dense(numClasses,activation=k.activations.softmax))

def addInputLayer(model,firstHidLayerDimm,InputDim,dropoutRate):
    assert dropoutRate < 1
    model.add(Dense(firstHidLayerDimm,input_dim=InputDim,activation=k.activations.relu,))
    model.add(Dropout(dropoutRate))

def addHidden(model,layerSizeList,dropoutRate):
    for size in layerSizeList:
        model.add(Dense(size,activation=k.activations.relu))
        model.add(Dropout(dropoutRate))

def createModel(x_train,numClasses,hiddenLayerSizeList,dropoutRate,optimizer):

    inputSize = x_train.shape[1]
    model = Sequential()

    addInputLayer(model,hiddenLayerSizeList[0],inputSize,dropoutRate)
    addHidden(model,hiddenLayerSizeList[1:],dropoutRate)
    addOutputLayer(model, numClasses)

    model.compile(loss=k.losses.categorical_crossentropy,metrics=['accuracy'],optimizer=optimizer)
    print(model.summary())
    return model

if __name__ == '__main__':

    xidx = 0
    yidx  =1

    c10All = k.datasets.cifar10.load_data()
    c10Train = c10All[0]
    c10Test = c10All[1]

    height = c10Test[xidx].shape[1]
    width = c10Test[xidx].shape[2]
    cch = c10Test[xidx].shape[3]

    c10X_train = c10Train[xidx].reshape(c10Train[xidx].shape[0],height*width*cch)
    c10X_train = skp.normalize(c10X_train,norm='l1')
    c10Y_train = c10Train[yidx].reshape(len(c10Train[yidx]))
    c10Y_train = tf.one_hot(c10Y_train,10)

    c10X_test= c10Test[xidx].reshape(c10Test[xidx].shape[0], height * width * cch)
    c10X_test = skp.normalize(c10X_test, norm='l1')
    c10Y_test = c10Test[yidx].reshape(len(c10Test[yidx]))
    c10Y_test = tf.one_hot(c10Y_test, 10)

    c100All = k.datasets.cifar100.load_data(label_mode="fine")
    c100Train = c100All[0]
    c100Test = c100All[1]

    c100X_train = c100Train[xidx].reshape(c100Train[xidx].shape[0], height * width * cch)
    c100X_train = skp.normalize(c100X_train, norm='l1')
    c100Y_train = c100Train[yidx].reshape(len(c100Train[yidx]))
    c100Y_train = tf.one_hot(c100Y_train, 10)

    c100X_test = c100Test[xidx].reshape(c100Test[xidx].shape[0], height * width * cch)
    c100X_test = skp.normalize(c100X_test, norm='l1')
    c100Y_test = c100Test[yidx].reshape(len(c100Test[yidx]))
    c100Y_test = tf.one_hot(c100Y_test, 10)




    layerSizes = [128,256,512]
    layerNums = [5,10,20]
    dropouts = [.2,.4,.6]
    optimizers = {'adam':k.optimizers.Adam(),
                  'SDG':k.optimizers.SGD()}
    modelTrainNum = 0


    c10df = pd.DataFrame([],columns=['layerSize','layerNum','dropout','optimizer','accuracy','executionTime'])
    c100df = pd.DataFrame([],columns=['layerSize','layerNum','dropout','optimizer','accuracy','executionTime'])

    print('Started c10')

    c10start_time = time.time()
    for layerSize in layerSizes:
        for layerNum in layerNums:
            for dropout in dropouts:
                for optimizerStr in optimizers.keys():

                    modelTrainNum = modelTrainNum + 1
                    optimizer = optimizers[optimizerStr]
                    modelStartTime = time.time()

                    print("now training model for c10: " +str(modelTrainNum) + " of " +
                          (str(len(layerSizes)*len(layerNums)*len(dropouts)*len(optimizers))))

                    #create the model and do the fitting here

                    c10M = createModel(c10X_train, 10, [layerSize for x in range(layerNum)], dropout, optimizer)
                    c10M.fit(c10X_train, c10Y_train, epochs=150, batch_size=500)
                    _, accuracy = c10M.evaluate(c10X_test, c10Y_test)

                    ###

                    modelExecutionTime =  time.time() - modelStartTime
                    print("Done training model for c10: total execution time: " + str(time.time() - c10start_time))
                    print("Model execution time: " + str(modelExecutionTime))
                    row = {'layerSize':layerSize,
                            'layerNum':layerNum,
                           'dropout':dropout,
                           'optimizer':optimizerStr,
                           'accuracy':accuracy,
                           'executionTime':modelExecutionTime}
                    c10df = c10df.append(row,True)

    c10df.to_csv('c10Results.csv')

    print('Started c100')

    c100start_time = time.time()
    modelTrainNum = 0

    for layerSize in layerSizes:
        for layerNum in layerNums:
            for dropout in dropouts:
                for optimizerStr in optimizers.keys():
                    modelTrainNum = modelTrainNum + 1
                    optimizer = optimizers[optimizerStr]
                    modelStartTime = time.time()

                    print("now training model for c100: " + str(modelTrainNum) + " of " +
                          (str(len(layerSizes) * len(layerNums) * len(dropouts) * len(optimizers))))

                    # create the model and do the fitting here

                    c100M = createModel(c100X_train, 100, [layerSize for x in range(layerNum)], dropout, optimizer)
                    c100M.fit(c100X_train, c100Y_train, epochs=150, batch_size=500)
                    _, accuracy = c100M.evaluate(c100X_test, c100Y_test)

                    ###

                    modelExecutionTime =  time.time() - modelStartTime
                    print("Done training model for c100: total execution time: " + str(time.time() - c100start_time))
                    print("Model execution time: " + str(modelExecutionTime))
                    row = {'layerSize': layerSize,
                           'layerNum': layerNum,
                           'dropout': dropout,
                           'optimizer': optimizer,
                           'accuracy': accuracy,
                           'executionTime': modelExecutionTime}
                    c100df = c100df.append(row, True)

    c100df.to_csv('c100Results.csv')



    # print('Accuracy: %.2f' % (accuracy * 100))