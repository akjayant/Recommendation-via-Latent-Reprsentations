#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:03:53 2020

@author: Ashish Kumar Jayant
@Title: PDS Assignment - 1 : Rating Prediction using Matrix Completion Techniques


#---------------MOVIE RATING PREDICTION PROBLEM (SIMILAR TO NETFLIX PRIZE)--------------
Data - 
(user,item,rating) - train + test

Goal - Train your model on train data and then predict rating for test user and item 
so as to minimise RMSE(true_rating,predicted_rating)

#-----------------------------STRATEGY USED FOR TRAINING--------------------------------
# MATRIX FACTORIZATION USING SGD IMPLEMENTED AS CLASS "Scratch_MF_SGD"
#  i.e, X (dimension : (m,n)) = U (dimension : (m,k)).transpose(V (dimension:(n,k)))
#  Cross validation on parameter 'k' and regularization parameter 'beta'
#---------------------------------------------------------------------------------------




"""

import sys
import numpy as np
#from sklearn.metrics import mean_squared_error
import pickle


#-------------AUXILLARY FUNCTIONS-----------------------------------------------

def make_matrix(user,item,rating,mask,list_user,list_item,m,n):
    
    if mask==False:
        A = np.zeros([m,n])
    elif mask==True:
        A = np.empty([m,n])
        A[:] = np.nan 
    for i in range(len(user)):
        index_user = list_user.index(user[i])
        index_item = list_item.index(item[i])
        A[index_user][index_item] = rating[i]
    return A

def predict(X_app,user,item,list_user,list_item,test_flag):
    if test_flag==True:
        approximated_matrix = open("/home2/e0268-56/a1/approximated_matrix_pkl.pkl","rb")
        X_app = pickle.load(approximated_matrix)
        
        list_user_pkl = open("/home2/e0268-56/a1/list_user_pkl.pkl","rb")
        list_user = pickle.load(list_user_pkl)
        
        list_item_pkl = open("/home2/e0268-56/a1/list_item_pkl.pkl","rb")
        list_item = pickle.load(list_item_pkl)
    
    predictions = []
    for i in range(len(user)):
        try:
            index_user = list_user.index(user[i])
            index_item = list_item.index(item[i])
            predictions.append(X_app[index_user][index_item])
        except:
            predictions.append(2.5)
    return predictions

#-----------SIMILAR TO 2 FOLD VALIDATION---------------------------------------
def train_validation_split(user,item,rating,ratio,policy):
    if policy==1:
        ratio = 1- ratio
        user_train = user[int(len(user)*ratio):]
        item_train = item[int(len(user)*ratio):]
        rating_train = rating[int(len(user)*ratio):]

        user_validation = user[:int(len(user)*ratio)]
        item_validation = item[:int(len(user)*ratio)]
        rating_validation = rating[:int(len(user)*ratio)]
    elif policy==2:
        user_train = user[:int(len(user)*ratio)]
        item_train = item[:int(len(user)*ratio)]
        rating_train = rating[:int(len(user)*ratio)]

        user_validation = user[int(len(user)*ratio):]
        item_validation = item[int(len(user)*ratio):]
        rating_validation = rating[int(len(user)*ratio):]   
    return user_train,item_train,rating_train,user_validation,item_validation,rating_validation

def rmse(true,pred):
    error = mean_squared_error(true,pred)
    rmse = np.sqrt(error)
    return rmse

#---Referred article https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf "Stochastic Gradient Descent"
    
class scratch_sgd_mf():
    
    def __init__(self,X,K,learning_rate, regularization,iters):
        self.X = X
        self.K = K
        self.lr = learning_rate
        self.regu = regularization
        self.iters = iters
        self.m = X.shape[0]
        self.n = X.shape[1]
        
    def fit(self):
        self.U = np.zeros([self.m, self.K])
        self.V = np.zeros([self.n, self.K])      
        self.bias_u = np.zeros(self.m)
        self.bias_i = np.zeros(self.n)
        self.mean_rating = np.mean(self.X[np.where(self.X != 0)])
        #Known entries        
        self.omega = [(i, j, self.X[i, j]) for i in range(self.m) for j in range(self.n) if self.X[i, j] > 0]
        
        #calling gradient descent sample by sample in omega i.e, SGD
        for i in range(self.iters):
            np.random.shuffle(self.omega)
            self.sgrad()
            
        print("fit done")        

    def sgrad(self):
 
        for user_index, item_index, rating in self.omega:
            lookup = self.lookup_rating(user_index, item_index)
            e = (rating - lookup)
            self.bias_u[user_index] += self.lr * (e - self.regu * self.bias_u[user_index])
            self.bias_i[item_index] += self.lr * (e - self.regu * self.bias_i[item_index])
            self.U[user_index, :] += self.lr * (e * self.V[item_index, :] - self.regu * self.U[user_index,:])
            self.V[item_index, :] += self.lr * (e * self.U[user_index, :] - self.regu * self.V[item_index,:])

    def lookup_rating(self, user, item):
   
        prediction = self.mean_rating + self.bias_u[user] + self.bias_i[item] + self.U[user, :].dot(self.V[item, :].T)
        return prediction

    def compute_matrix(self):
        return self.U.dot(self.V.T) + self.mean_rating + self.bias_u[:,np.newaxis] + self.bias_i[np.newaxis:,] 






#------------------------------GETTING TEST USERS,ITEMS ------------------------
def driver(train_flag):

#-----TESTING-----------------------
    if train_flag==0:    
        f_test = open(sys.argv[1],'r')
        file_test = []
        for x in f_test:
            file_test.append(x)
        file_test = file_test[1:]
        
        user_test=[]
        item_test=[]
        for i in file_test:
            user_test.append(int(i.split('\t')[0].strip()))
            item_test.append(int(i.split('\t')[1].strip()))
        f_test.close()
        
        predictions = predict(0,user_test,item_test,0,0,True)
        
        out_test = open(sys.argv[2],'w')
        for i in predictions:
            if i>=0.5 and i<=5.0:
                out_test.write(str(round(i,5))+'\n')
            elif i<0.5:
                out_test.write(str(0.5)+'\n')
            elif i>5.0:
                out_test.write(str(5.0)+'\n')     
#----TRAINING----------------------        
    elif train_flag==1:
        f = open(sys.argv[1],'r')
        file = []
        for x in f:
            file.append(x)
        #print(file)
        file = file[1:]
        user=[]
        item=[]
        rating=[]
        for i in file:
            user.append(int(i.split('\t')[0].strip()))
            item.append(int(i.split('\t')[1].strip()))
            rating.append(float(i.split('\t')[2].strip()))

        # Unique users and unique items
        set_user = set(user)
        m = len(set_user)
        set_item = set(item)
        n = len(set_item)
        # for indexing
        list_user = list(set_user)
        list_item = list(set_item) 
        #cross validation
        
        k_param = [50,60,70,80,90,100]
        rmse_k_dict = {}
        for k in k_param:
            mean_validation_rmse = 0
            for policy in [1,2]:
                user_train,item_train,rating_train,user_validation,item_validation,rating_validation = train_validation_split(user,item,rating,0.75,policy)
                X_train = make_matrix(user_train,item_train,rating_train,False,list_user,list_item,m,n)
                model_mf = scratch_sgd_mf(X_train, K=k, learning_rate=0.01, regularization=0.1, iters=100)
                model_mf.fit()
                SGD_M = model_mf.compute_matrix()
                SGD_pred = predict(SGD_M,user_validation,item_validation,list_user,list_item,False)
                print("validation error = "+str(rmse(rating_validation,SGD_pred)))
                mean_validation_rmse += rmse(rating_validation,SGD_pred)
            #performance of k over two folds
            rmse_k_dict[k] = mean_validation_rmse/2
        print(rmse_k_dict)
        best_k = min(rmse_k_dict, key=rmse_k_dict.get)
        #---Train on full matrix with best 'k'
        X_full = make_matrix(user,item,rating,False,list_user,list_item,m,n)
        model_mf_best = scratch_sgd_mf(X_full, K=best_k, learning_rate=0.01, regularization=0.1, iters=100)
        model_mf_best.fit()
        SGD_M = model_mf_best.compute_matrix()
        print("Model_trained!")
        print(SGD_M)
        
        approximated_matrix = open("approximated_matrix_pkl.pkl","wb")
        pickle.dump(SGD_M,approximated_matrix)
        
        list_user_pkl = open("list_user_pkl.pkl","wb")
        pickle.dump(list_user,list_user_pkl)
        
        list_item_pkl = open("list_item_pkl.pkl","wb")
        pickle.dump(list_item,list_item_pkl)
        
# For training flag = 1

#driver(1)

# For testing flag = 0
driver(0)
#-------------------------------------------------------------------------------
    
    

   





































    

#---------------------------------------------------------------------------------------    
