
# coding: utf-8
import os
os.environ["CUDA__DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from keras.models import Model
from keras.datasets import cifar10
import keras.backend as K

from time import sleep
from math import ceil
from collections import Counter
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from random import shuffle
import numpy as np
import pandas as pd


def join_image_paths(imagePath):
    # Joins image filenames with rest of whole path
    paths = []

    # Cycles through first image/directory in input path
    for oneDeep in os.listdir(imagePath):
        # Joins whole path
        pathOne = os.path.join(imagePath, oneDeep)

        # If path is found to be a file (image), paths list is appended
            # This will therefore mean that all of the contents within the 
            # input directory are images
        if os.path.isfile(pathOne):
            paths.append(pathOne)
        # Else (if path is found to be a directory)
        else:
            # Cycles through images in directory found in input directory
                # This will therefore mean that there are folders of images
                # within the input directory
            for twoDeep in os.listdir(pathOne):
                pathTwo = os.path.join(pathOne, twoDeep)
                paths.append(pathTwo)
        
    # Shuffles paths list and outputs
    shuffle(paths)
    return paths

def remove_some_eth(df, label, reduce, poss_labels):
    # Removes some examples from specified eth to try and match up category example sizes
    
    # Gets base DF, all rows not including labels to reduce
    base = df[df['ethnicity'] != label].reset_index(drop=True)
    
    # Gets rows including labels to reduce then reduces it
    eth_to_drop = df[df['ethnicity'] == label]
    eth = eth_to_drop.sample(frac=reduce).reset_index(drop=True)
    
    # Concats reduced labels with all other labels split previously
    new_df = pd.concat([base, eth]).sample(frac=1).reset_index(drop=True)
    
    # Outputs new label counts
    print('Ethnicity %s reduced... new counts...' % label)
    for l in sorted(set(new_df['ethnicity'].tolist())):
        print('%s: %s' % (l, new_df[new_df['ethnicity']==l].shape[0]))
        
    return new_df

def replace_image_names(paths, DF):
    # Data cleaning... alters image names for more standardized format
    
    # Creates new DF for standardized image paths and ethnicities
    print('Replacing filenames in DF with full paths')
    new_df = pd.DataFrame(columns=('path', 'ethnicity'))

    # Replaces image cells to only include "sub.[SUBNUMwxyz]" from cssff_ethLabels.csv file
    df = format_subs(DF)

    # Gets deduplicated list of possible ethincities
    poss_labels = sorted(set(DF['ethnicity'].tolist()))
    
    # Enumerates through paths getting subject number
    for i, onePath in enumerate(paths):
        # Tries to parse by one format type... If this fails it tries parsing by other format type
        try:
            # Tries to parse format of "BESTCROP-zzzz.1-sub.[SUBNUMwxyz]_cssfacefacts_cssfacefacts.jpg"
                # Where SUBNUMwxyz is the subject number
            int(onePath.split('/')[-1].split('_')[0].split('.')[-1])
            fileSub = '-'.join(onePath.split('/')[-1].split('-')[2:]).split('_')[0]
        except:
            # Tries to parse format of "sub_[SUBNUMwxyz]_frame_XX_get_face.jpg"
                # where SUBNUMwxyz is the subject number
            int(onePath.split('/')[-1].split('_')[1])
            fileSub = '.'.join(onePath.split('/')[-1].split('_')[:2])
            
        # Gets ethnicity for file and loads params into new_df
        fileEth = df[df['image']==fileSub]['ethnicity'].iloc[0]
        new_df.loc[i] = [onePath, fileEth]
        
    # Prints out label counts
    for oneLabel in poss_labels:
        print('%s: %s' % (oneLabel, new_df[new_df['ethnicity'] == oneLabel].shape[0]))
        
    print('\n')
    
    return new_df

def format_subs(DF):
    DF['image'] = DF['image'].apply(lambda c: c.split('_')[0])
    return DF

def split_sets(DF, tr_path, val_path, ts_path, labels_path):
    # Gets training, validation and testing sets
    
    # Splits up sets to 60/20/20 split
    train, validate, test = np.split(DF.sample(frac=1).reset_index(drop=True), [int(0.6*len(DF)),
                                                                               int(0.8*len(DF))])
    
    # Resets all indicies
    train, validate, test = train.reset_index(drop=True), validate.reset_index(drop=True), test.reset_index(drop=True)
    
    # Writes data to files for later quicker loading
    train.to_csv(tr_path, index=False)
    validate.to_csv(val_path, index=False)
    test.to_csv(ts_path, index=False)
    DF.to_csv(labels_path, index=False)
    
    # Returns sets
    return train, validate, test

def getOneHot(arr, key):
    # Formats classes to oneHot encoding
    print('Processing one-hot encoding...')
    output = np.zeros((arr.shape[0], len(key)))
    for i, elem in enumerate(arr):
        output[i][key.index(elem)] = 1
        
    return output

def conv_oneHot(DF):
    # Converts oneHot encoded classes from string to actual list
        # This is needed when loading in files with preprocessed data of
        # oneHot encoding
    DF['ethnicity'] = DF['ethnicity'].apply(lambda c: str_to_oneHot(c))
    
    return DF
    
def str_to_oneHot(string):
    # Converts string oneHot encoded sequence to list
    string = string.replace('[', '').replace(']', '').replace(',', '')
    out_list = np.fromstring(string, dtype=np.float, sep=' ')
    
    return out_list

def load_sets(imagePath='data/cssff-crops', ethPath='data/cssff_ethLabels.csv',
              tr_path='data/resnetEthAlgo/load_files/eth_train.csv',
              val_path='data/resnetEthAlgo/load_files/eth_validation.csv', 
              test_path='data/resnetEthAlgo/load_files/eth_testing.csv', 
              labels_path='data/resnetEthAlgo/load_files/eth_labels.csv', proc_choice=1):
    # Either processes images and data or loads preprocessed data from file

    # Gets all non-processed image file and subject data
    ethData = pd.read_csv(ethPath, index_col=False)
    # Gets possible class labels (de-duplicated)
    poss_labels = sorted(set(ethData['ethnicity'].tolist()))
    
    # Gets all paths to images for training
    cropPaths = join_image_paths(imagePath)
    
    # If-else for processing or loading pre-processed data
        # 1 - Process data
        # 2 - Load pre-processed data from CSVs
    if proc_choice==1:
        print('Processing paths with first choice... This may take some time...')
        # Replace image names to be more standardized then reduce Caucasian set by 40% to even out classes
        ethData = remove_some_eth(replace_image_names(cropPaths, ethData), label='Caucasian', reduce=0.6,
                                 poss_labels=poss_labels)
        # Replace ethnicity column with the oneHot equivalents
        ethData['ethnicity'] = list(getOneHot(np.array(ethData['ethnicity'].tolist()), poss_labels))
        
        # Split up data into training, validation and testing set... also saves sets to CSVs for later loading
        training_set, validation_set, testing_set = split_sets(ethData, tr_path, val_path, test_path, labels_path)
        
    elif proc_choice==2:
        print('Processing paths with second choice... Loading paths from files...')
        # Load sets from CSVs
        training_set = conv_oneHot(pd.read_csv(tr_path))
        validation_set = conv_oneHot(pd.read_csv(val_path))
        testing_set = conv_oneHot(pd.read_csv(test_path))
        ethData = pd.read_csv(labels_path)


    # Output final sizes    
    print('\nTraining, validation and testing sets loaded...')
    print('Sizes:\nTraining: %s\nValidation: %s\nTesting: %s' % (training_set.shape[0], validation_set.shape[0],
                                                                testing_set.shape[0]))

    # Return all relevant data    
    return training_set, validation_set, testing_set, ethData, poss_labels


def load_images(image_paths, size=224):
    all_images = np.zeros((len(image_paths), size, size, 3))
    
    for i, _path in enumerate(image_paths):
        # Read image
        image = plt.imread(_path)
        # Resize
        img = cv2.resize(image, (size, size))
        # Normalize
        img = np.divide(img, [255., 255., 255.])
        # Append
        all_images[i] = img
        
    return all_images

def printO(string, filename, header=True, custom_header=False, new_line=True, terminal_output=True):
    # Function for printing updates to log incase notebook session is closed on local machine
    
    # If-else for clearing file or appending contents
    if string == 'CLEARFILE':
        f = open(filename, 'w')
        
        # If-else for creating header at beginning of empty log
        if custom_header:
            fill = custom_header
        elif header:
            fill = 'Training began at %s\n\n' % datetime.now()
        else:
            fill = ''
        f.write(fill)
        f.close()
        print('User output file %s cleared...' % filename.split('/')[-1])
        
    else:
        if terminal_output:
            print(string)
        f = open(filename, 'a')
        if new_line:
            string += '\n'
        f.write(string)
        f.close()
        
def getMaxBatch(DF, max_set):
    # Function for reducing input DF and putting reduced rows into separate DF

    # Create new DF and reset index of input DF
    current_batch = pd.DataFrame(columns=list(DF))
    DF = DF.reset_index(drop=True)
    
    # Get correct size for amount to pull from input DF
        # Makes sure no indexing issues are run into
    if max_set > DF.shape[0]:
        max_set = DF.shape[0]+1
        
    # Pulls data from input DF into new DF
    current_batch = DF.loc[:max_set, :]
    
    # Drops previously pulled data
    DF = DF.drop(DF.index[:max_set])
    
    # Returns both DFs and resets both indexes
    return DF.reset_index(drop=True), current_batch.reset_index(drop=True)

def getBatch(batchDF, image_max):
    # Gets a batch of images and labels by...
        # Reducing input DF by image_max and getting a current batch of data
        # Loads images at paths specified in current_batch
        # Grabs labels of images loaded in current_batch
        # Returns reduced input DF, loaded images and labels
    batchDF, current_batch = getMaxBatch(batchDF, image_max)
            
    X_tr = load_images(current_batch['path'].tolist())
    Y_tr = current_batch['ethnicity'].tolist()
    
    return batchDF, X_tr, Y_tr

def mostCommonIncorrectPred(incorrect_dict):
    # Figures out and outputs most common incorrect prediction made for each class
    outStr = '\n\nActual Ethnicity: Most Common Misprediction\n'
    outCSV = 'ACT_LABEL,' + ','.join(classes) + '\n'

    # Iterates through possible classes
    for key in classes:
        amount = []
        outCSV += key + ','
        # Iterates through possible classes to look for predictions made
        for lookFor in classes:
            # Gets counts of mispredictions for specific class
            amount.append(incorrect_dict[key].count(lookFor))
            outCSV += str(amount[-1]) + ','
            
        # Outputs class and its associated largest misprediction
        outCSV += '\n'
        outStr += '%s: %s\n' % (key, classes[np.argmax(amount)])
        
    return outStr, outCSV

def postBatchProcessing(c, wholeDF, tempDF, image_max, ticker, file_to_write, stage='training'):
    # Function for post batch updates to log files and updating some small variables
    c+=1 
    printO('\n%s - At iteration %s of %s for batches' % (stage, c, ceil(wholeDF.shape[0]/image_max)),
           filename=file_to_write)
    printO('Estimated time for this epoch\'s %s completion: %s\n' % (stage,
                            ((datetime.now()-ticker)*ceil(tempDF.shape[0]/image_max))),
           filename=file_to_write)
    ticker = datetime.now()
    
    return c, ticker

# Learning rate scheduler
# lr to be reduced based on number of epochs
def lr_schedule(epoch):

    lr = 0.015
    if epoch > 40:
        lr *= 0.01

    printO('Learning rate: %s' % lr, filename=progress_log)
    return lr

# A function to build layers for the Resnet:
    # 1. Conv
    # 2. Batch normalization
    # 3. Activation
def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization

    # Returns
        x (tensor): tensor as input to the next layer
    """
    # Convolution operation
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

    x = inputs
    x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


def resnet_create(input_shape, depth, num_classes=10, stack_size=6):
    """
    First stack does not change the size
    Later, at the beginning of each stack, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stack 0: 224x224, 16
    stack 1: 112x112, 32
    stack 2:  56x56,  64
    stack 3:  28x28,  128
    stack 4:  14x14,  256
    stack 5:  7x7,  512
    GlobalAveragePooling 7x7, 512 -> 1x1, 512
    Flatten 1x1, 512 -> 512

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44)')

    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    # Instantiate the stack of residual units
    for stack in range(stack_size):
        for res_block in range(num_res_blocks):
            strides = 1

            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None, batch_normalization=False)

            # Add skip connection
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)

        num_filters *= 2 # Increase number of filter

    # Add average pooling then flatten
    x = AveragePooling2D(pool_size=7)(x)
    y = Flatten()(x)

    # Add softmax output
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # Instantiate model and output
    model = Model(inputs=inputs, outputs=outputs)
    return model

def getAcc(model, X_set, Y_set, incorrect_dict):
    # Gets accuracy of model by doing predictions on input set
    preds = model.predict(X_set)
    if not incorrect_dict:
        incorrect_dict = {key: [] for key in classes}
        
    # Counts the amount of correct predictions and logs any incorrect predictions
    correct = 0
    for i, prediction in enumerate(preds):
        if np.argmax(prediction) == np.argmax(Y_set[i]):
            correct += 1
        else:
            actu_eth = classes[np.argmax(Y_set[i])]
            pred_eth = classes[np.argmax(prediction)]
            incorrect_dict[actu_eth].append(pred_eth)
            
    return correct/len(preds), incorrect_dict

def midTrainTest(model, X_batch, Y_batch, incorrect_dict, acc, batches, file_to_write, batchTime, 
                             stage='training', total_batches=0):
    # Mid training test for between batch fitting or prediction
        # Can be for training sequence or validation

    # Gets accuracy and incorrect prediction counts then updates accuracy and loss and updates log
    a, incorrect_dict = getAcc(model, X_batch, Y_batch, incorrect_dict)
    acc.append(a)
    loss.append(model.evaluate(X_batch, Y_batch, verbose=0))
    printO("%s - Batch : %s" % (stage, batches), filename=file_to_write)
    printO("%s - of batch %s" % (stage, total_batches), filename=file_to_write)
    printO("Estimated time for %s batch completion: %s" % (stage, ((datetime.now()-batchTime)/10)*
                                                        (total_batches-batches)),
          filename=file_to_write)
    
    batchTime = datetime.now()
    
    return acc, incorrect_dict, batchTime

def updateEpochMetrics(epoch_metrics, metrics_file, acc, allAcc, loss, epoch, stage, file_to_write):
    # Updates epoch metrics at end of training or validation stage for an epoch

    # Updates log with epoch, mean and loss
    new_line = True
    if stage == 'train':
        new_line = False
        met = '%s,%s,%s,' % (epoch+1, np.mean(acc)*100, np.mean(loss))
    elif stage == 'validation':
        met = '%s,%s' % (np.mean(acc)*100, np.mean(loss))
    
    # Updates epoch metrics dictionary and updates logs
    epoch_metrics[stage].append([np.mean(acc)*100, np.mean(loss)])
    printO(met, filename=metrics_file, new_line=new_line)
    printO('%s mean of accuracy and loss for this epoch: %s' % (stage, epoch_metrics[stage][-1]),
          filename=file_to_write)
    
    return epoch_metrics

def completeEpoch(epoch_metrics, metrics_file, acc, allAcc, loss, incorrect_dict, incorrectPreds_file,
                      epoch_tick, epochs, epoch, file_to_write, models_base):
    # Sequence of events to occur at the end of an epoch

    # All accuracy list appended and then string and CSV output for logs is found for incorrect predictions
    allAcc.append(np.mean(acc)*100)
    s, outCSV = mostCommonIncorrectPred(incorrect_dict)
    printO(outCSV, filename=incorrectPreds_file)
    
    # Checks if model should be saved
    if max(allAcc) == allAcc[-1]:
        # Gets path for model and model log
        model_path = os.path.join(models_base, 'ResNet-Ethnicity_Acc-%.2f_.h5' % allAcc[-1])
        log_path = os.path.join(models_base, 'OUTLOG_ResNet-Ethnicity_Acc-%.2f_.txt' % allAcc[-1])

        # Removes and previously recorded models
        for _file in os.listdir(models_base):
        	_file = os.path.join(models_base, _file)
        	os.remove(_file)

        # Clears (initiates) log file then outputs log contents and saves model
        printO('CLEARFILE', filename=log_path)
        outStr = 'LOG INFO FOR ETHNICITY TRAINING BEST MODEL FOUND \n\nEPOCH: %s\nACCURACY: %s\nLOSS: %s' % (
            epoch+1, allAcc[-1], np.mean(loss))
        outStr += s
        printO(outStr, filename=log_path)
        
        model.save(model_path)
        
    # Outputs epoch metrics and gives time estimations
    printO('Accuracy of epoch: %.2f' % (np.mean(acc)*100), filename=file_to_write)
    printO('Time of epoch: %s' % (datetime.now()-epoch_tick), filename=file_to_write)
    printO('As of %s, estimated time remaining for training/validation: %s' % (datetime.now(),
                            (datetime.now()-epoch_tick)*(epochs-(epoch+1))),
          filename=file_to_write)
    epoch_tick = datetime.now()
    
    return allAcc, epoch_tick


# Get all image paths
image_path = 'data/cssff-crops'
#imagePaths = join_image_paths(image_path)

# Creates log file paths and base folder paths for models and logs
base_models = 'data/resnetEthAlgo/models'
base_logs = 'data/resnetEthAlgo/logs'
progress_log = os.path.join(base_logs, 'ResNet-printout.txt')
metrics_path = os.path.join(base_logs, 'ResNet-ethModelMetrics.csv')
incorrect_preds_path = os.path.join(base_logs, 'ResNet-incorrectPredMetrics.csv')

# Clears logs
printO('CLEARFILE', filename=progress_log)
printO('CLEARFILE', filename=metrics_path, custom_header='EPOCH,TR_ACC,TR_LOSS,VAL_ACC,VAL_LOSS\n')
printO('CLEARFILE', filename=incorrect_preds_path, header=False)

# Get training, validation and testing sets. Also gets full data and de-duplicated class list
tr_set, val_set, ts_set, eth_data, classes = load_sets(proc_choice=2)
num_classes = len(classes)

# Sets up image data generator for training
imageGen = ImageDataGenerator(horizontal_flip=True)#, vertical_flip=True)

# Creating model with depth of 8 and stack size of 6
depth = 8
stack_size=6
# Input image dimensions.
input_shape = [224, 224, 3]
# Creates resnet model with specified depth and stack size
model = resnet_create(input_shape=input_shape, depth=8, num_classes=len(classes), stack_size=stack_size)

# Compiles model and outputs architecture
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr_schedule(0), momentum=0.8), metrics=['accuracy'])
#printO(model.summary(), filename=progress_log, terminal_output=False)
print('Model compiled...')

# Some hyperparameters for training
epochs=50
image_max=3200
batch_size=128

# Initializes epoch metric and timing variables
allAcc = []
ticker = datetime.now()
epoch_tick = datetime.now()
epoch_metrics = {'train': [], 'validation': []} # Metrics list for loss and accuracy to be stored in

# Enters try-except statement in case anything happens, output logs will be updated with error
try:
    # Iterates through epochs
    for i in range(epochs):
        # Updates log and prints epoch
        printO('\n' + '='*30 + '\n' + 'Epoch %s of %s' % (i+1, epochs) + '\n' + '='*30 + '\n', 
                           filename=progress_log)

        # Iteration counter (c), stage accuracy list (acc), stage loss list (loss), temp training set (temp_train)
            # and incorrect predictions keeper reset/initialized
        c = 0
        acc = []
        loss = []
        temp_train = tr_set
        incorrect_dict = False

        # Update learning rate if needed
        update_lr = lr_schedule(i)
        K.set_value(model.optimizer.lr, update_lr)

        # Keeps iterating until temp training set has been reduced to a size of 0
        while temp_train.shape[0] > 0:
            # Reduces temp_train set and puts reduced rows into current batch variables (X_tr, Y_tr)
            temp_train, X_tr, Y_tr = getBatch(temp_train, image_max)

            # Resets/initializes timing and counting variables
            batchTime = datetime.now()
            batches=0
            tr_total_batches = len(X_tr) // batch_size

            # Uses data generator to load images and labels from X_tr and Y_tr in sizes of batch_size
            for X_batch, Y_batch in imageGen.flow(X_tr, Y_tr, batch_size=batch_size):
                # Fits current batches and increments batch counter
                model.fit(X_batch, Y_batch, verbose=0)
                batches += 1

                # Breaks image data generator if batch count has been reached for the current set
                if batches >= len(X_tr) / batch_size:
                    break
                # Updates logs and terminal output every 10 batches
                elif batches % 10 == 0:
                    acc, incorrect_dict, batchTime = midTrainTest(model, X_batch, Y_batch, incorrect_dict, acc,
                                                      batches, progress_log, batchTime, 'training', tr_total_batches)

            # Updates logs and terminal output with post batch metrics
            c, ticker = postBatchProcessing(c, tr_set, temp_train, image_max, ticker, progress_log)

        # Updates logs and terminal output with training metrics
        epoch_metrics = updateEpochMetrics(epoch_metrics, metrics_path, acc, allAcc, loss, i, 'train', progress_log)

        # Iteration counter (c), stage accuracy list (acc), stage loss list (loss), temp training set (temp_cal)
            # and incorrect predictions keeper reset/initialized
        incorrect_dict = False
        temp_val = val_set
        ticker = datetime.now()
        c = 0
        acc = []

        # Keeps iterating until temp validation set has been reduced to a size of 0
        while temp_val.shape[0] > 0:
            # Reduces temp_val set and puts reduced rows into current batch variables (X_val, Y_val)
            temp_val, X_val, Y_val = getBatch(temp_val, image_max)

            # Resets/initializes timing and counting variables
            batchTime = datetime.now()
            batches = 0
            val_total_batches = len(X_val) // batch_size

            # Uses data generator to load images and labels from X_tr and Y_tr in sizes of batch_size
            for X_batch, Y_batch in imageGen.flow(X_val, Y_val, batch_size=batch_size):
                # Makes predictions, gets accuracy of predictions and records it then increments batch counter
                a, incorrect_dict = getAcc(model, X_batch, Y_batch, incorrect_dict)
                acc.append(a)
                batches += 1

                # Breaks image data generator if batch count has been reached for current set
                if batches >= len(X_val) / batch_size:
                    break
                # Updates logs and terminal output every 10 seconds
                elif batches % 10 == 0:
                    acc, incorrect_dict, batchTime = midTrainTest(model, X_batch, Y_batch, incorrect_dict, acc,
                                                      batches, progress_log, batchTime, 'validation', val_total_batches)

            # Updates logs and terminal output with post batch metrics
            c, ticker = postBatchProcessing(c, val_set, temp_val, image_max, ticker, progress_log, 'validation')

        # Updates logs and terminal output with training metrics
        epoch_metrics = updateEpochMetrics(epoch_metrics, metrics_path, acc, allAcc, loss, i, 'validation',
                                          progress_log)

        # Further updates logs and terminal output with end of epoch metrics
        allAcc, epoch_tick = completeEpoch(epoch_metrics, metrics_path, acc, allAcc, loss, 
                        incorrect_dict, incorrect_preds_path, epoch_tick, epochs, i, progress_log,
                                          base_models)
            
# If an error is occured, log is updated.
except Exception as e:
    printO('MODEL TRAINING FAILURE\n\nERROR:\n%s' % e, filename=progress_log)