**Creating Train and Test Datasets from Annotaion Files**


```py

# Import the python dataset class
from createDataset import createDataset 

# Define the objects in list with the id begins with 0
class_object = {0 : 'background' , 1 : 'objectClass1' , 2:'objectClass2'}

# Dataset object
train_set = createDataset()

# Makes data ready for use
train_set.prepare_dataset('pizza/annots' , class_object , test_size=30)

# Assing train and test data
(trainY , trainX) , (testY , testX ) = (train_set.trainY , train_set.trainX ) , (train_set.testY , train_set.testX ) 

```
