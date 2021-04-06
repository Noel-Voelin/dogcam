Dog Cam with Tensorflow
==============

## General Information
First attempt of creating a simpel model that detects my dog and does something upon detection

### Pre Conditions
Python3 (preferably anaconda) installed and all the required models added. Further, Tensorflow should be installed and configured according to the host machiene (either GPU or CPU optimised) 

### Setup
For creation of the model and the pickle files, create folders that are called according to the catrgories defined in DogTensor.py
once the folders are created (for example dog and human) fill the folders with a sufficient amount of sample data (around 10k images per category would be good)

### Test the model you created
the DogTensor.py contains a main method which evaluates a static "test" image against the model. Just add an image with the name "test.jpg" to the project and run the main method of the TensorDog Class. If all he data mentioned in the setup is created, all required files should be created and the model should be trained automatically before the test.jpg is predicted
