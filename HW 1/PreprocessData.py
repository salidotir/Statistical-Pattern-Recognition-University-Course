#!/usr/bin/env python
# coding: utf-8

# # Preprocessing data

# In[2]:


# imports

import numpy as np
import pandas as pd


# In[5]:


class Dataset:
    def __init__(self, data_train_file, data_test_file, output_column_name='y', normalization_method='zero_mean_unit_var'):
        """
        self.x_train                 ->     vector-x0 + data_x_train_normalized
        self.x_test                  ->     vector-x0 + data_x_test_normalized
        
        self.x_train_without_x0      ->     data_x_train_normalized
        self.x_test_without_x0       ->     data_x_test_normalized
        
        self.y_train                 ->     y_train
        self.y_test                  ->     y_test
        """
        
        # read csv file using pandas
        self.data_train = pd.read_csv(data_train_file)
        self.data_test = pd.read_csv(data_test_file)
        
        # export y_train & y_test and convert to numpy array
        self.y_train = self.data_train[output_column_name].to_numpy(dtype='float64')
        self.y_test = self.data_test[output_column_name].to_numpy(dtype='float64')
        
        # remove y from x_train & x_test
        self.x_train_without_x0 = self.data_train.drop([output_column_name], axis=1).to_numpy(dtype='float64')
        self.x_test_without_x0 = self.data_test.drop([output_column_name], axis=1).to_numpy(dtype='float64')
        
        # required variables for model
        self.shape_of_x_train_without_x0 = self.x_train_without_x0.shape
        self.shape_of_x_test_without_x0 = self.x_test_without_x0.shape
        
        self.normalize(normalization_method)
        self.x_train, self.x_test = self.add_vector_x0()
            
        
    """
    Normalizing data improves the convergence of learning model and causes that smaller features also be able to affect the model parameters.
    """
    def normalize(self, normalization_method):
        if normalization_method == 'none':
            print("No normalization.")
            return
        
        if normalization_method == 'zero_mean_unit_var':
            print("zero-mean & unit_variance normalization.")
            self.x_train_without_x0 = self.zero_mean_unit_variance(self.x_train_without_x0)
            self.x_test_without_x0 = self.zero_mean_unit_variance(self.x_test_without_x0)
            
            
        if normalization_method == 'scale_0_1':
            print("scaling to [0, 1] normalization.")
            self.x_train_without_x0 = self.scaling_between_0_1(self.x_train_without_x0)
            self.x_test_without_x0 = self.scaling_between_0_1(self.x_test_without_x0)
     
    
    def scaling_between_0_1(self, numpy_array):
        '''
        Scaling
        '''
        normed_numpy_array = (numpy_array - numpy_array.min(axis=0)) / (numpy_array.max(axis=0) - numpy_array.min(axis=0))
        return normed_numpy_array


    def zero_mean_unit_variance(self, numpy_array):
        '''
        Standardization
        '''
        normed_numpy_array = (numpy_array - numpy_array.mean(axis=0)) / numpy_array.std(axis=0)
        return normed_numpy_array

    def add_vector_x0(self):
        # create a vector of ones for x0 and add to first of x_train & x_test
        # to add x0_vector of ones to coloumns, we should use np.c_
        # to add x0_vector of ones to rows, we should use np.r_
        
        x0_vector_train = np.ones(shape=self.y_train.shape)
        x0_vector_test = np.ones(shape=self.y_test.shape)
        x_train = np.c_[x0_vector_train, self.x_train_without_x0]
        x_test = np.c_[x0_vector_test, self.x_test_without_x0]
        return x_train, x_test


# In[24]:


# dataset = Dataset('Data-Train.csv', 'Data-Test.csv', 'y', normalization_method='zero_mean_unit_var')

# print(dataset.x_train.mean(axis=0))
# print(dataset.x_train.std(axis=0))

# print(dataset.x_test.mean(axis=0))
# print(dataset.x_test.std(axis=0))

