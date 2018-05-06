# Weld_EthnicityDetection
Institution: Rowan University\
Term: Spring 2018\
Course: Topics in Machine Learning and Data Mining\
Professor: Dr. Rasool\
Project By: Robert Weld\
Project Description: Final project for classification of subject ethnicity within an image

## DISCLAIMER
All data used for the training of this algorithm is proprietary and cannot be uploaded to the git repository. Data not used for training has been uploaded to the 'images' directory as an example of what was used and the data corresponding to these images can be found in the 'subData' directory.

## SETUP
Main python dependencies for this project include...
1. Keras
2. TensorFlow (Set as Keras Backend)
3. OpenCV2
4. Matplotlib
5. Pandas
6. Numpy
7. dlib

Dlib dependencies for this project include...
1. shape\_predictor\_68\_face\_landmarks.dat - This file can be downloaded at [this link](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat)
* This file must be input to the main project folder with face_cropper.py for the facial cropping script to work.

When adding new data...
1. Add image(s) to 'images/original' directory
2. Run 'face_cropper.py -i images/original' from terminal in local repo main folder
3. Append data/ethLabels.csv folder with a new row with new images' data

## ADDITIONAL INFORMATION
1. Before running code for model creation, the data cleaning methods should be checked over to avoid any runtime errors. 
2. As it is easily seen, the VGG16 model is an iPython notebook while the ResNet model is trained in a python script. This is because the server used for inital training with VGG16 was taken offline and another server without a running jupyter kernal had to be used. Both training models will work, though, after the data cleaning methods are checked to avoid cleaning errors.
