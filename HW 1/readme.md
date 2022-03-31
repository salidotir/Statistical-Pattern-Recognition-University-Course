# Regression using Closed-Form Solution vs. Gradient Descent Solution
Implemented two diffrent regression solution on Data-Train & Data-Test

### Dataset
* In PreprocessData file, we read data_train & data_test and do the preprocessing operations.

* By creating an instance of Dataset class, you can access x_train, y_train, x_test and y_test.

* Also, it is possible to choose normalization method among 'none', 'scale_0_1' and 'zero_mean_unit_variance'.


### Regression Class
* In this class, all functions required for both solutions is implemented such as predict(), plot() and report().

* fit() function has an empty body and will be completed in each of the below class as their method works.

* Also objective funciton(MSE) is defined as below in Regression class:
> <img src="https://user-images.githubusercontent.com/35997721/144423680-79f821f1-4fba-4701-93e2-720412f34f7b.png" width="200">


### Closed Form Class 
This class inherits from Regression class and 𝜃 is calculated as:

> <img src="https://user-images.githubusercontent.com/35997721/144424240-bc6f9597-c068-41c2-9294-840ac704e999.png" width="200">


### Gradient Descent Class
Gradient descent is used to minimize objective(or cost) function by repetitively moving in the direction of steepest descent in each epoch (epoch means pass over the training dataset). Here we implement Batch Gradient Descent.

> <img src="https://user-images.githubusercontent.com/35997721/144425032-c23e9d83-bb95-4767-9f5d-b3bb3690fcb4.png" width="200">

### Result

> <img src="https://user-images.githubusercontent.com/35997721/144425357-b3c800e8-4d7b-4bbc-ab6e-f69f2155fc5f.png" width="400">
