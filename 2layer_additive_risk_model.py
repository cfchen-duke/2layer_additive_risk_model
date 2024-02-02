# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:33:40 2018

@author: Yaron.Shaposhnik
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn import datasets
from sklearn import linear_model
import pandas as pd
import scipy

from helpers import split_train_test

import copy



def compute_prob1(w_,c_,X):
    return(1/(1+np.exp(-(np.dot(X, w_.reshape(-1,1))+c_))))

def compute_prob0(w_,c_,X):
    return(1-compute_prob1(w_,c_,X))


counter = -1
def logistic_loss(coef, params):
    global counter
    counter+=1
    if ('DISPLAY_PROGRESS' in params) and (counter%params['DISPLAY_PROGRESS']==0):
        print('Fitting model; iteration',counter,'out of',params['MAX_ITER'])
    #print(counter, coef)
    X = params['X']
    y = params['y']    
    n,p = X.shape
    
    if params['intercept']:
        assert(len(coef)==p+1)
        w = coef[:p]
        c = coef[p]
    else:
        assert(len(coef)==p)
        w = coef
        c = 0
    
    res = 0
    for i in range(n):
        # http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        res+=params['C']*np.log(1+np.exp(-y[i]*(np.dot(w,X[i,:])+c)))
    if params['penalty']=='l1':
        res += np.sum(np.abs(w))
    elif params['penalty']=='l2':
        res += 0.5*np.dot(w,w)
    else:
        raise Exception('Unsupported regulatization')
    return(res)
    
    
    
        
class LogisticRegressionConstrained(BaseEstimator, ClassifierMixin):
    def __init__(self, params = None):
        if params is None:
            params = init_default_params()
        self.params_ = params
        self.load_model_ = False
    
    
    def fit(self, X, y):
    
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
    
        self.X_ = X
        self.y_ = y

        self.params_['X']=X
        self.params_['y']=y
        self.func = lambda x: logistic_loss(x, self.params_)

        n,p = X.shape
        
        bounds = []
        for i in range(p):
            if i in self.params_['POSITIVE_COEF']:
                bounds.append((0,np.inf))
            elif i in self.params_['NEGATIVE_COEF']:
                bounds.append((-np.inf,0))
            else:
                bounds.append((-np.inf,np.inf))
        if self.params_['intercept']:
            bounds.append((-np.inf,np.inf))             # add bounds for the offset parameter
        
        x0 = self.params_['x0']
        if x0 is None:
            if self.params_['intercept']:
                x0 = np.zeros(p+1)
            else:
                x0 = np.zeros(p)
        #x, nfeval, rc = scipy.optimize.fmin_tnc(self.func, x0,approx_grad=True, bounds=bounds, maxfun=10**6)        
        #x, nfeval, rc = scipy.optimize.fmin_tnc(self.func, x0,approx_grad=True, bounds=bounds, maxfun=10**6)        
        x, f, d = scipy.optimize.fmin_l_bfgs_b(self.func, x0,approx_grad=True, bounds=bounds, maxfun=self.params_['MAX_ITER'])        
        global counter
        counter = -1


        #x = self.params_['temp'] #!!!!!
        
        if self.params_['intercept']: 
            self.w_, self.c_ = x[:p], x[p]       
            self.coef_, self.intercept_ = x[:p].reshape((1,-1)), x[p:p+1]
        else:
            self.w_ = x       
            self.coef_ = x.reshape((1,-1))
            self.c_ = 0
            self.intercept_ = 0
        
        return self
    
    
    def predict(self, X):
        if self.load_model_ == False:
            # Check is fit had been called
            check_is_fitted(self, ['X_', 'y_'])
    
        # Input validation
        X = check_array(X)

        n1,p1 = X.shape   
        self.probs_ = compute_prob1(self.w_,self.c_,X).reshape(-1)
        res = ((self.probs_>=0.5).astype(int))*2-1                
        
        return(res)
    

    def predict_proba(self, X):
        if self.load_model_ == False:
            # Check is fit had been called
            check_is_fitted(self, ['X_', 'y_'])
    
        # Input validation
        X = check_array(X)

        n1,p1 = X.shape   
        
        p0 = compute_prob0(self.w_,self.c_,X)
        p1 = compute_prob1(self.w_,self.c_,X)
        return(np.concatenate([p0,p1],axis=1))

    def load_model_from_memory(self, coef, intercept):
        self.load_model_ = True
        self.w_ = coef[0] # coef is a 1-by-many matrix 
        self.coef_ = coef
        self.c_ = intercept
        self.intercept_ = intercept



def init_default_params():
    return({'penalty': 'l2', 'POSITIVE_COEF': [], 'NEGATIVE_COEF': [], 'C': 1.0,
            'X': None, 'y': None, 'x0': None, 'intercept': True, 
            'MAX_ITER': 10**5, 'DISPLAY_PROGRESS': 100})

    
monotonicity_var_list = [3, 4, 5, 6, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 33, 
                         34, 35, 36, 42, 43, 44, 45, 50, 51, 52, 53, 66, 67, 68, 
                         69, 73, 74, 75, 76, 81, 82, 83, 84, 89, 90, 91, 92, 97, 
                         98, 99, 100, 114, 115, 116, 117, 129, 130, 131, 132, 
                         138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 
                         157, 170, 171, 172, 173]
subscale_num_attributes = [8, 22, 8, 32, 32, 24, 24, 16, 8, 8]
num_subscales = len(subscale_num_attributes)

class TwoLayerConstrainedLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, params = None):
        if params is None:
            params = init_default_params()
        self.params_ = params
    
    def fit(self, X, y):
        monotonicity_var_list = self.params_['POSITIVE_COEF']
        subscale_clfs = []
        subscale_start_attribute = 0
        for subscale_index, num_attributes in enumerate(subscale_num_attributes):
            subscale_attributes = list(range(subscale_start_attribute,
                                             subscale_start_attribute + num_attributes))
            subscale_monotonicity = set(subscale_attributes).\
                                        intersection(set(monotonicity_var_list))
            subscale_monotonicity = list(subscale_monotonicity)
            subscale_monotonicity = [var_index - subscale_start_attribute \
                                     for var_index in subscale_monotonicity]
            X_subscale = X[:,subscale_start_attribute:\
                             (subscale_start_attribute+num_attributes)]
            params = copy.deepcopy(self.params_)
            params['POSITIVE_COEF'] = subscale_monotonicity
            clf_subscale = LogisticRegressionConstrained(params)
            clf_subscale.fit(X_subscale, y)
            subscale_clfs.append(clf_subscale)
            if subscale_index == 0:
                subscale_scores = clf_subscale.predict_proba(X_subscale)[:, 1].reshape((-1, 1))
            else:
                subscale_scores = np.hstack((subscale_scores,
                                             clf_subscale.predict_proba(X_subscale)[:, 1].reshape((-1, 1))))
            subscale_start_attribute = subscale_start_attribute + num_attributes
        
        self.subscale_clfs_ = subscale_clfs
        
        params = copy.deepcopy(self.params_)
        params['POSITIVE_COEF'] = list(range(num_subscales))
        clf_output = LogisticRegressionConstrained(params)
        clf_output.fit(subscale_scores, y)
        
        self.clf_output_ = clf_output
        
        return self
    
    def predict(self, X):
        subscale_scores = self._build_subscale_scores(self, X)
        clf_output = self.clf_output_
        
        return clf_output.predict(subscale_scores)
    
    def predict_proba(self, X):
        subscale_scores = self._build_subscale_scores(self, X)
        clf_output = self.clf_output_
        
        return clf_output.predict_proba(subscale_scores)
    
    def predict_and_explain(self, X, k=3):
        subscale_scores = self._build_subscale_scores(self, X)
        clf_output = self.clf_output_
        clf_output_weight = clf_output.coef_
        
        weighted_subscale_scores = np.multiply(subscale_scores, clf_output_weight)
        
        predictions = clf_output.predict(subscale_scores).astype(float).reshape((-1, 1))
        
        top_subscales = np.argmax(np.multiply(weighted_subscale_scores, predictions), axis=1)
        
        subscale_start_attribute_list = [0]
        for num_attributes in subscale_num_attributes:
            subscale_start_attribute_list.append(subscale_start_attribute_list[-1]+num_attributes)
        
        X_top_subscales = [X[i,subscale_start_attribute_list[top_subscale]:subscale_start_attribute_list[top_subscale+1]] \
                           for (i,top_subscale) in enumerate(top_subscales)]
        subscale_clfs = self.subscale_clfs_
        top_subscale_clf_weights = [subscale_clfs[top_subscale].coef_[0] \
                                    for top_subscale in top_subscales]
        
        weighted_features_all = [np.multiply(x, top_subscale_clf_weights[i]) \
                                 for (i,x) in enumerate(X_top_subscales)]
        
        predictions = predictions.reshape(-1)
        
        explanations = [np.argsort(weighted_features*predictions[i])[::-1] \
                        for (i,weighted_features) in enumerate(weighted_features_all)]
        explanations = [explanation + subscale_start_attribute_list[top_subscales[i]] \
                        for (i,explanation) in enumerate(explanations)]
        
        topk_explanations = np.array([explanation[:k] for explanation in explanations])
        topk_values = np.array([X[i, topk_explanations_i] \
                                  for (i,topk_explanations_i) in enumerate(topk_explanations)],
                                  dtype=np.int)
        predictions = predictions.astype(int)
        
        return predictions, clf_output.predict_proba(subscale_scores), \
               topk_explanations, topk_values
    
    def find_similar_cases(self, X, predictions, k, topk_explanations, topk_values,
                           X_, y_, num_cases=1):
        X_ = np.hstack((X_, np.array(list(range(len(X_)))).reshape(-1, 1)))
        y_ = y_.astype(int)
        X_pos = X_[y_==1]
        X_neg = X_[y_==-1]
        similar_cases = []
        degrees_of_similarity = []
        top_similarity_scores = []
        num_similar_cases_same_cls = []
        num_similar_cases_opp_cls = []
        
        for (i,x) in enumerate(X):
            if predictions[i] == 1:
                X_same = X_pos
                X_opp = X_neg
            else:
                X_same = X_neg
                X_opp = X_pos
            X_same_ = X_same[:, topk_explanations[i]].astype(int)
            X_opp_ = X_opp[:, topk_explanations[i]].astype(int)
            
            similarities_same_cls = np.sum((X_same_==topk_values[i]).astype(int), axis=1)
            similar_cases_sorted = np.argsort(similarities_same_cls)[::-1]
            similar_cases_id = similar_cases_sorted[:num_cases]
            similar_cases_ = X_same[similar_cases_id, -1].astype(int)
            similar_cases.append(similar_cases_)
            
            degrees_of_similarity_sorted = np.sort(similarities_same_cls)[::-1]
            degrees_of_similarity_ = degrees_of_similarity_sorted[:num_cases]
            degrees_of_similarity.append(degrees_of_similarity_)
            
            top_similarity_score = np.amax(similarities_same_cls)
            top_similarity_scores.append(top_similarity_score)
            
            num_similar_cases_same_cls_ = np.bincount(similarities_same_cls,
                                                      minlength=k+1)[top_similarity_score]
            num_similar_cases_same_cls.append(num_similar_cases_same_cls_)
            
            similarities_opp_cls = np.sum((X_opp_==topk_values[i]).astype(int), axis=1)
            num_similar_cases_opp_cls_ = np.bincount(similarities_opp_cls,
                                                     minlength=k+1)[top_similarity_score]
            num_similar_cases_opp_cls.append(num_similar_cases_opp_cls_)
            
        similar_cases = np.array(similar_cases)
        degrees_of_similarity = np.array(degrees_of_similarity)
        top_similarity_scores = np.array(top_similarity_scores)
        num_similar_cases_same_cls = np.array(num_similar_cases_same_cls)
        num_similar_cases_opp_cls = np.array(num_similar_cases_opp_cls)
        
        return similar_cases, degrees_of_similarity, top_similarity_scores,\
               num_similar_cases_same_cls, num_similar_cases_opp_cls
    
    def _build_subscale_scores(self, X):
        subscale_start_attribute = 0
        for subscale_index, num_attributes in enumerate(subscale_num_attributes):
            X_subscale = X[:,subscale_start_attribute:\
                             (subscale_start_attribute+num_attributes)]
            clf_subscale = self.subscale_clfs_[subscale_index]
            if subscale_index == 0:
                subscale_scores = clf_subscale.predict_proba(X_subscale)[:, 1].reshape((-1, 1))
            else:
                subscale_scores = np.hstack((subscale_scores,
                                             clf_subscale.predict_proba(X_subscale)[:, 1].reshape((-1, 1))))
            subscale_start_attribute = subscale_start_attribute + num_attributes
        
        return subscale_scores
    
    def save_model_weights(self, filename='2layerLRC.npz'):
        weights_and_biases = {}
        for subscale_index in range(num_subscales):
            clf_subscale = self.subscale_clfs_[subscale_index]
            weights_and_biases['weight_subscale_%d' % subscale_index] = clf_subscale.coef_
            weights_and_biases['bias_subscale_%d' % subscale_index] = clf_subscale.intercept_
        clf_output = self.clf_output_
        weights_and_biases['weight_output'] = clf_output.coef_
        weights_and_biases['bias_output'] = clf_output.intercept_
        np.savez(filename, **weights_and_biases)

    def load_model_weights(self, filename):
        weights_and_biases = np.load(filename)
        self.subscale_clfs_ = []
        for subscale_index in range(num_subscales):
            subscale_coef = weights_and_biases['weight_subscale_%d' % subscale_index]
            subscale_intercept = weights_and_biases['bias_subscale_%d' % subscale_index]
            clf_subscale = LogisticRegressionConstrained()
            clf_subscale.load_model_from_memory(subscale_coef, subscale_intercept)
            self.subscale_clfs_.append(clf_subscale)
        output_coef = weights_and_biases['weight_output']
        output_intercept = weights_and_biases['bias_output']
        clf_output = LogisticRegressionConstrained()
        clf_output.load_model_from_memory(output_coef, output_intercept)
        self.clf_output_ = clf_output
        
        return self

def save_data_split(X_train, y_train, X_test, y_test):
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

def load_train_and_test_data_with_random_split(filename, save_split=False):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    data_train, data_test = split_train_test(data, test_size=0.2)
    
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    y_train = y_train.astype(int)
    y_train[y_train == 0] = -1
    #y_train = np.reshape(y_train, [len(y_train), 1])
    y_train = y_train.astype(float)

    print(X_train.shape)

    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    y_test = y_test.astype(int)
    y_test[y_test == 0] = -1
    #y_test = np.reshape(y_test, [len(y_test), 1])
    y_test = y_test.astype(float)
    
    print(X_test.shape)
    
    if save_split:
        save_data_split(X_train, y_train, X_test, y_test)
    
    return X_train, y_train, X_test, y_test
    
def train_and_test_on_random_split(path_to_split='./'):    
    # Load dataset
    dataset_path = '../dataset/full_discrete/'
    filename = dataset_path + 'full.csv'
    if not(path_to_split is None):
        try:
            X_train = np.load('X_train.npy')
            y_train = np.load('y_train.npy')
            X_test = np.load('X_test.npy')
            y_test = np.load('y_test.npy')
        except:
            X_train, y_train, X_test, y_test = \
                load_train_and_test_data_with_random_split(filename=filename,
                                                           save_split=False)
    else:
        X_train, y_train, X_test, y_test = \
            load_train_and_test_data_with_random_split(filename=filename,
                                                       save_split=False)

    # Optimization parameters
    MAX_ITER=10**3
    C = 1
    
    # Run Logistic regression using SK-Learn
    print('------------ SK-Learn ------------')    
    clf = linear_model.LogisticRegression(penalty='l2', C=C, solver='lbfgs',
                                          max_iter=MAX_ITER, intercept_scaling=False)
    clf.fit(X_train, y_train)
    print('Training accuracy:', np.mean(clf.predict(X_train)==y_train))
    print('Test accuracy:', np.mean(clf.predict(X_test)==y_test))
    print('Coefficients: w,c', clf.coef_, clf.intercept_)
    print('Predictions:')
    print(clf.predict_proba(X_test[:5,:]))
    print(clf.predict(X_test[:5,:]))
    print(y_test[:5])
    print(clf.predict(X_test[:5,:])==y_test[:5])



    #X_train[X_train == 0] = -1
    #X_test[X_test == 0] = -1
    MAX_ITER=2000
    C = 1
    
    # Run Logistic regression using SK-Learn    
    print('\n\n------------ Customized ------------')        
    clf2 = LogisticRegressionConstrained({'penalty': 'l2',
                                          #'POSITIVE_COEF': list(range(X_train.shape[1])),
                                          'POSITIVE_COEF': monotonicity_var_list,
                                          'NEGATIVE_COEF': [],
                                          'C': C,
                                          'x0': None,
                                          'intercept': True,
                                          'MAX_ITER': MAX_ITER,
                                          'DISPLAY_PROGRESS': 100})
    clf2.fit(X_train, y_train)
    print('Training accuracy:', np.mean(clf2.predict(X_train)==y_train))
    print('Test accuracy:', np.mean(clf2.predict(X_test)==y_test))
    print('Coefficients: w,c', clf2.coef_, clf2.intercept_)
    print('Predictions:')
    print(clf2.predict_proba(X_test[:5,:]))
    print(clf2.predict(X_test[:5,:]))
    print(y_test[:5])
    print(clf2.predict(X_test[:5,:])==y_test[:5])
    print(min(clf2.predict(X_test)))
    
    print('\n\n------------ Customized: two-layer ------------')        
    clf3 = TwoLayerConstrainedLogisticRegression({'penalty': 'l2',
                                                  #'POSITIVE_COEF': list(range(X_train.shape[1])),
                                                  'POSITIVE_COEF': monotonicity_var_list,
                                                  'NEGATIVE_COEF': [],
                                                  'C': C,
                                                  'x0': None,
                                                  'intercept': True,
                                                  'MAX_ITER': MAX_ITER,
                                                  'DISPLAY_PROGRESS': 100})
    clf3.fit(X_train, y_train)
    print('Training accuracy:', np.mean(clf3.predict(X_train)==y_train))
    print('Test accuracy:', np.mean(clf3.predict(X_test)==y_test))
    #print('Coefficients: w,c', clf3.coef_, clf3.intercept_)
    print('Predictions:')
    print(clf3.predict_proba(X_test[:5,:]))
    print(clf3.predict(X_test[:5,:]))
    print(y_test[:5])
    print(clf3.predict(X_test[:5,:])==y_test[:5])
    print(min(clf3.predict(X_test)))
    clf3.save_model_weights(filename='2layerLRC_split.npz')
    
    # Export results
#    if 0:
#        df = pd.DataFrame(X)
#        df.columns = ['X[%d]'%i for i in range(X.shape[1])]
#        df['y']=y
#        df['y_sk']=clf.predict(X)
#        df['y_yaron']=clf2.predict(X)
#        df.to_csv('temp.csv')
    
def train_on_entire_dataset():
    dataset_path = '../dataset/full_discrete/'
    filename = dataset_path + 'full.csv'
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    
    X_train = data[:, 1:]
    y_train = data[:, 0]
    y_train = y_train.astype(int)
    y_train[y_train == 0] = -1
    #y_train = np.reshape(y_train, [len(y_train), 1])
    y_train = y_train.astype(float)

    print(X_train.shape)
    
    # Optimization parameters
    MAX_ITER=10**3
    C = 1
    
    # Run Logistic regression using SK-Learn
    print('------------ SK-Learn ------------')    
    clf = linear_model.LogisticRegression(penalty='l2', C=C, solver='lbfgs',
                                          max_iter=MAX_ITER, intercept_scaling=False)
    clf.fit(X_train, y_train)
    print('Training accuracy:', np.mean(clf.predict(X_train)==y_train))
    print('Coefficients: w,c', clf.coef_, clf.intercept_)
    print('Predictions:')
    print(clf.predict_proba(X_train[:5,:]))
    print(clf.predict(X_train[:5,:]))
    print(y_train[:5])
    print(clf.predict(X_train[:5,:])==y_train[:5])



    #X_train[X_train == 0] = -1
    #X_test[X_test == 0] = -1
    MAX_ITER=2000
    C = 1
    
    # Run Logistic regression using SK-Learn    
    print('\n\n------------ Customized ------------')        
    clf2 = LogisticRegressionConstrained({'penalty': 'l2',
                                          #'POSITIVE_COEF': list(range(X_train.shape[1])),
                                          'POSITIVE_COEF': monotonicity_var_list,
                                          'NEGATIVE_COEF': [],
                                          'C': C,
                                          'x0': None,
                                          'intercept': True,
                                          'MAX_ITER': MAX_ITER,
                                          'DISPLAY_PROGRESS': 100})
    clf2.fit(X_train, y_train)
    print('Training accuracy:', np.mean(clf2.predict(X_train)==y_train))
    print('Coefficients: w,c', clf2.coef_, clf2.intercept_)
    print('Predictions:')
    print(clf2.predict_proba(X_train[:5,:]))
    print(clf2.predict(X_train[:5,:]))
    print(y_train[:5])
    print(clf2.predict(X_train[:5,:])==y_train[:5])
    print(min(clf2.predict(X_train)))
    
    print('\n\n------------ Customized: two-layer ------------')        
    clf3 = TwoLayerConstrainedLogisticRegression({'penalty': 'l2',
                                                  #'POSITIVE_COEF': list(range(X_train.shape[1])),
                                                  'POSITIVE_COEF': monotonicity_var_list,
                                                  'NEGATIVE_COEF': [],
                                                  'C': C,
                                                  'x0': None,
                                                  'intercept': True,
                                                  'MAX_ITER': MAX_ITER,
                                                  'DISPLAY_PROGRESS': 100})
    clf3.fit(X_train, y_train)
    print('Training accuracy:', np.mean(clf3.predict(X_train)==y_train))
    #print('Coefficients: w,c', clf3.coef_, clf3.intercept_)
    print('Predictions:')
    print(clf3.predict_proba(X_train[:5,:]))
    print(clf3.predict(X_train[:5,:]))
    print(y_train[:5])
    print(clf3.predict(X_train[:5,:])==y_train[:5])
    print(min(clf3.predict(X_train)))
    clf3.save_model_weights(filename='2layerLRC_entire_dataset.npz')
    np.save('2layerLRC_predictions.npy', clf3.predict(X_train))

def test_load_model(filename):
    dataset_path = '../dataset/full_discrete/'
    dataset_filename = dataset_path + 'full.csv'
    data = np.genfromtxt(dataset_filename, delimiter=',', skip_header=1)
    
    X = data[:, 1:]
    
    clf = TwoLayerConstrainedLogisticRegression()
    clf.load_model_weights(filename)
    subscale_scores = clf._build_subscale_scores(X)
    
    np.save('subscale_scores_entire_dataset.npy', subscale_scores)


if __name__ == "__main__" and 1:
    train_on_entire_dataset()
    #test_load_model(filename='2layerLRC_entire_dataset.npz')
    
    











