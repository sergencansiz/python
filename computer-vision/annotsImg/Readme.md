**Creating Train and Test Datasets from Annotaion Files**


```py
from createDataset import createDataset as cd 

class_object = {0 : 'pizza' , 1 : 'background'}
train_set = cd()
train_set.prepare_dataset('pizza/annots' , class_object , test_size=30)
(trainY , trainX) , (testY , testX ) = (train_set.trainY , train_set.trainX ) , (train_set.testY , train_set.testX ) 
```
