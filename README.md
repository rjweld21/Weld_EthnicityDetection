# Weld_EthnicityDetection
Final project for Rowan University Spring 2018 Machine Learning course with Dr.
Rasool.

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

When adding new data...
1. Add image(s) to 'images/original' directory
2. Run 'face_cropper.py -i images/original' from terminal in local repo main folder
3. Append data/ethLabels.csv folder with a new row with new images' data
