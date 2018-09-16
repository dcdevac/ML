"""
This is a Python library intended to be used as a framework for ML regression and classification of tabluar data.  
It comprises the following classes:
dc_project :    The class used to host the overall project with dataset, models, a metric etc.
dc_dataset:     A class used to hold the different stages of data (raw, processed), the splits between data features and labels.
                Implements some data wrangling methods such as removing outliers based on quantile tolerances.
dc_model:       A wrapper class for a range of different models (Keras, SKLearn, XGBoost, LGBM) supporting Regressor, Binary Classifier 
                and Multiclass Classifier.  Wraps fit, predict, feature importance, save, load etc.
                This may be redundant as Keras, XGBoost, LGBM all appear to have implemented the SKLearn API somewhat.
dc_helper:      A small collection of static helper functions.


Some parts are not fully tested or finished
"""


from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras import models, layers
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb

from collections import defaultdict

import math

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn import clone

import pickle



class mtype(Enum):
    """
    Enum to label the 4 different types of models (Keras, SKLearn, XGBoost, LGBM))
    """
    SKL = 0
    XG = 1
    LGBM = 2
    KERAS = 3 


 
class dc_model:
    """
    Wrapper class to create a standard interface for different types of models (Keras, SKLearn, XGBoost, LGBM) 
    and implement commonnly used model features (feature importance, hyperparameter search etc.)
    """
    # Note:  Perhaps better implemented as a base class with derived classes for Regressor, Binary Classifier, Multiclass Classifier, 
    # but I started this way and continued.

    ID_counter = 0   # to yield unique model names
    
    def __init__(self, model_pointer, num_classes = 0, name = None):
        self.num_classes = num_classes  # 0 = Regressor, 1 = Binary Classifier, >1 = Multiclass Classifier
        self.model = model_pointer      # pointer to the underlying model
        self.__get_mtype()              # Set the model type
        self.is_trained = False
        #  Different models may require removal of unimportant features. input_cols_to_ignore allows user 
        #   to specify certain columns to ignore when fitting the processed df
        self.input_cols_to_ignore = []  
        
        if name is None:
            self.name = str(self.mtype) + "_" + str(dc_model.ID_counter)
            dc_model.ID_counter += 1
        else:
            self.name = name
        self.__hist = None              #optional parameter to hold the hist for a Keras fit 


    def __get_mtype(self):
        #  Assume everything except Keras, XGBoost and LGBM is a SKLearnModel
        if isinstance(self.model, (XGBClassifier, XGBRegressor) ):
            self.mtype = mtype.XG
        elif isinstance(self.model, (lgb.Booster, lgb.LGBMClassifier, lgb.LGBMRegressor, lgb.LGBMModel, lgb.LGBMRanker) ):
            self.mtype = mtype.LGBM
        elif isinstance(self.model, models.Sequential ):
            self.mtype = mtype.KERAS
        else:
        	self.mtype = mtype.SKL


    def save(self, filename):
        if self.mtype in (mtype.XG, mtype.SKL):
            pickle.dump(self.model, open(filename, "wb"))
        elif self.mtype == mtype.LGBM:
            self.model.save_model(filename)
        elif self.mtype == mtype.KERAS:
            self.model.save(filename+".h5")
    

    def load(self, model_type, filename):
        if self.mtype in (mtype.XG, mtype.SKL):
            self.model = pickle.load(open(filename, "rb"))
        elif self.mtype == mtype.LGBM:
            self.model = lgb.LGBMModel
            self.model = lgb.Booster(model_file = filename)
        elif self.mtype == mtype.KERAS:
            # TODO
            pass
            

    def fit(self, x_data_passed, y_data, p_epochs = 10, p_batch_size = 128):
        x_amended = x_data_passed.drop(self.input_cols_to_ignore, axis = 1, inplace = False)
        if self.mtype == mtype.SKL:
            self.model.fit(x_amended, y_data)
        elif self.mtype == mtype.XG:
            self.model.fit(x_amended, y_data)
        elif self.mtype == mtype.LGBM:
            lgb_train = lgb.Dataset(x_amended, y_data)
            self.model.fit(x_amended, y_data)
            # TODO Unimplemented
            pass
        elif self.mtype == mtype.KERAS:
            self.__hist = self.model.fit(x_amended, y_data, epochs = p_epochs, batch_size = p_batch_size, verbose = 0)
        self.is_trained = True


    def __str__(self):
        return self.name

    
    def predict(self, x_data):
        if self.num_classes == 0:
            return self.model.predict(x_data)
        elif self.num_classes == 1:
            return np.round(self.model.predict(x_data)).astype(int) 
        else:
            #TODO check this
            return np.argmax(self.predict_proba(x_data) )

    
    def predict_proba(self, x_data):
        """
        For classifiers, returns the prediction probablity (0-1) for each class
        """
        if self.num_classes == 0:
            print("Cannot call predict_proba on a regressor.  Use predict instead.")
        elif self.num_classes == 1:
            return [[float(1-x), float(x)] for x in list(self.model.predict(x_data))]
        else:
            # TODO check this
            return self.model.predict(x_data)


    def hyperparameter_search(self, x_data, y_data):
        # TODO Perhaps enable some common searches
        pass


    # TODO: verify that this is untrained/initialized weights for all mtypes 
    def get_deep_copy(self):
        """
        This creates an untrained copy of the model structure
        """
        if self.mtype == mtype.SKL:
            return clone(self.model)
        elif self.mtype == mtype.XG:
            if isinstance(self.model, XGBRegressor):
                return XGBRegressor(**self.model.get_params())
            if isinstance(self.model, XGBClassifier):
                return XGBClassifier(**self.model.get_params())
        elif self.mtype == mtype.LGBM:
            # TODO
            pass
        elif self.mtype == mtype.KERAS:
            # TODO Consider whether ref is to KerasNN wrapper or the model itself.  Probably latter.
            #return KerasNN(self.model.neuronList, self.model.dropout, self.model.epochs)
            pass


    def get_params(self):
        if self.mtype == mtype.SKL:
            return self.model.get_params()
        elif self.mtype == mtype.XG:
            pass
        elif self.mtype == mtype.LGBM:
            pass
        elif self.mtype == mtype.KERAS:
            return self.model.summary()


    def feature_importance(self):
        """
        Returns a list of feature importances for all models that support it.  
        List is normalized to sum to 1.0
        """
        if self.mtype in (mtype.XG, mtype.SKL):
            return self.model.feature_importances_ / self.model.feature_importances_.sum()
        elif self.mtype == mtype.LGBM:
            return self.model.feature_importance() / self.model.feature_importance().sum()
        elif self.mtype == mtype.KERAS:
            print("Feature importance not supported for Keras models.")
            return None






class dc_dataset:

    def __init__(self, df_raw = None, target_col_name = None):
        # self.raw
        # self.target_col_name     ...Listing these 2 class members here for readability
        self.processed = None
        self.x_train = None
        self.y_train = None
        self.x_partial_train = None
        self.y_partial_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None 
        #  Note no y_test.  This is unknown
        if df_raw is not None:
            self.load_train(df_raw, target_col_name)


    def load_train(self, df_train, target_col_name):
        assert isinstance(df_train, pd.DataFrame), f"{str(df_train)} is not a Pandas DataFrame"
        assert target_col_name in df_train.columns, f"{target_col_name} is not in the columns of {df_train}"
        self.target_col_name = target_col_name
        self.raw = df_train
        if self.raw[self.target_col_name].isna().sum() > 0:
            print (f"WARNING:  Training labels contains {self.raw[self.target_col_name].isna().sum()} NAN values")
        self.raw["Test"] = 0
        dc_helpers.move_col_right(self.raw, self.target_col_name)


    def load_test(self, df_test):
        assert isinstance(df_test, pd.DataFrame), f"{str(df_test)} is not a Pandas DataFrame"
        assert self.raw is not None, "Need to load training data before test data" 
        # ensure train has all same columns as test less the target col 
        assert set(self.raw.columns) - set(df_test.columns) == {self.target_col_name, "Test"}, \
               f"Column mismatch between train and test data.\n\nTrain contains {self.raw.columns}\n\nTest contains {df_test.columns}"
        df_test["Test"] = 1
        df_test[self.target_col_name] = -999 # the defalt "dummy" value
        self.raw = pd.concat((self.raw, df_test), axis = 0, sort=False)


    def guess_num_classes(self):
        """
        This method guessess the number of classification classes for the dataset 
        based on the number of unique values in the target column
        """
        assert self.raw is not None, "Need to load training data before calling guess_num_classes" 
        unique = len(self.raw[self.target_col_name].unique() )
        if unique == 2:
            print ("Guess binary classification with HIGH confidence.")
            return 2
        elif unique > 10 and self.raw[self.target_col_name].dtype in (np.dtype("float32"), np.dtype("float64")):
            temp_conf = "HIGH" if unique > 30 else "MEDIUM"
            print (f"There are {unique} float values.  Guess regression with {temp_conf} confidence.")
            return 0
        elif unique <= 10:
            print (f"There are {unique} targets.  Guess multiclass classification with HIGH confidence.")
            return unique
        else: # More than 10 unique values, but not a float.  
            print (f"There are {unique} targets.  Guess multiclass classification with LOW confidence")
            return unique


    def remove_outliers(self, column_list, tolerance, upper_only = False, lower_only = False):
        """
        This will remove upper and lower outliers from df based on a tolerance from 0 to 1 (in practice make close to 0).  
        All upper and lower items within a column within the given tolerance quantile are dropped from the df
        """
        temp = self.raw.shape[0]
        if not isinstance(column_list,(list, tuple)): # Allows for a single col to be passed
            column_list = [column_list]
        outlier_dict = dc_dataset.__build_outlier_dict(self, tolerance, column_list, upper_only, lower_only)
        print (outlier_dict)
        dc_dataset.__remove_based_on_dict(self, outlier_dict)
        temp = temp - self.raw.shape[0]
        if self.raw.shape[0] > 0:
            print ("Observations Removed:\t%i\nPercenage Removed:\t%f%%" % (temp, 100*temp/ self.raw.shape[0] ) )
        else:
            print ("No observations removed")


    def __remove_based_on_dict(self, outlier_dict):
        """
        This will remove outliers from df based on a dict in following format {'Feature': (-15.5, 8.5)}
        """
        for col_name in outlier_dict:
            to_drop = self.raw[ ( (self.raw[col_name] <= outlier_dict[col_name][0]) | 
            (self.raw[col_name] >= outlier_dict[col_name][1]) ) & (self.raw["Test"] == 0) ].index
            if len(to_drop > 0):
                self.raw.drop(to_drop, axis = 0, inplace=True)



    def __clamp_based_on_dict(self, outlier_dict):
        """
        This will hard clamp outliers from df based on a dict in following format {'Feature': (-15.5, 8.5)}  
        i.e. all values outside extents will be set to extents.
        """
        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        for col_name in outlier_dict:
            self.raw = self.raw.apply(lambda x: clamp(x, outlier_dict[col_name][0], outlier_dict[col_name][1]))


    def __build_outlier_dict(self, tolerance, column_list = None, upper_only = False, lower_only = False):
        """
        Builds a dictionary list of outliers.  tolerance = quantile distance (e.g. 5 = 5% - 95%)
        """
        assert not (upper_only == lower_only == True), "upper_only and lower_only should not both be true."
        column_list = self.raw.columns if column_list is None else column_list
        outlier_dict = {}
        for i in column_list:
            # if excluded, then just set artificially high or low tolerances, but leave a value in dictionary
            lo_tolerance = self.raw[i].quantile(tolerance) if upper_only == False \
                           else self.raw[i].quantile(0) - 999.0
            hi_tolerance = self.raw[i].quantile(1-tolerance) if lower_only == False \
                           else self.raw[i].quantile(1) + 999.0
            outlier_dict[i]=(lo_tolerance, hi_tolerance)
        return outlier_dict  


    def process_raw(self, dummy_list_index, normalize_list_index, drop_list_index, logify_list_index, logify_add = 0.00001):
        self.processed = self.raw.copy()
        # convert col indices into named string indexing, because dropping & dummyizing cols will change indexing
        to_dummyize = [self.processed.columns[i] for i in dummy_list_index ]
        to_normalize = [self.processed.columns[i] for i in normalize_list_index ]
        to_logify = [self.processed.columns[i] for i in logify_list_index ]
        to_drop = [self.processed.columns[i] for i in drop_list_index ]
        # Step 1: Logify
        for col in to_logify:
            self.processed[col] = np.log(self.processed[col] + logify_add)
        # Step 2: Normalize
        for col in to_normalize:
            self.processed[col] -= self.processed[col].mean()
            self.processed[col] /= self.processed[col].std()
        # Step 3: Create dummies
        self.processed = pd.get_dummies(self.processed, columns = to_dummyize, drop_first = True)
        # Step 4: Drop unused
        self.processed.drop(to_drop, axis = 1, inplace = True)
        # Step 5: Move target to rightmost column (for convenience & readability)
        temp = self.processed[self.target_col_name]
        self.processed.drop(labels=[self.target_col_name], axis=1, inplace = True)
        self.processed[self.target_col_name] = temp
        # Step 6:  Processed is complete.  Now split out x,y test,train
        self.x_train = self.processed[self.processed["Test"] == 0].iloc[:,:-1]
        self.y_train = self.processed[self.processed["Test"] == 0].iloc[:,-1]
        self.x_test  = self.processed[self.processed["Test"] == 1].iloc[:,:-1]


    def set_validation_data(self, val_proportion, random_state = 42):
        assert self.x_train is not None, "Need to set training data first"
        assert val_proportion >= 0 and val_proportion < 1, \
               f"val_proportion = {val_proportion}\nNeeds to be between 0 and 1"
        self.x_partial_train, self.x_val, self.y_partial_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=val_proportion, random_state=random_state)
        #  UNUSED:  Used skikitlearn auto method instead
        # # OldTODO Shuffle 
        # num_val_samples = len(self.data.x_train) * val_proportion
        # self.x_val = self.x_train[:num_val_samples]
        # self.y_val = self.y_train[:num_val_samples]
        # self.x_partial_train = self.x_train[num_val_samples:]
        # self.y_partial_train = self.y_train[num_val_samples:]







class dc_project:
    """
    This is a class for an entire ML project comprising a dataset and models.  
    Goal is to support all the following typical steps: (NOTE: Not all implemented)
        Load data
        • Load train
        • Load test
        • Identify primary metric
        • -999 for test and add Test col
        • Concatenate
        Data wrangling & feature engineering
        • Feature assessment - dtypes
        • Compare test values with training values 
        • Deal with missing values - consider dropping versus filling (data size matters)
        • Deal with outliers - remove or squash/clamp.  Consider test e.g. could you remove/clamp all values more than one standard deviation away from extremes in test
        • Deal with fragmented categorical data
        • Feature engineering
        Process data
        • Standardize, logify, drop
        • Create X & Y
        Add models
        • Identify candidate models (cross-fold)
        • Do grid search on promising models
        • Assess feature importance and finalize feature list: perhaps removing some features.  
            ○ Question:  should you actually drop less important features - better generalization?
        • Finalize final model list
        Train final models
        • Train final models on all data
        • TODO Collate predictions in tables
        • TODO Add averaging/ensemble
        • TODO Assess results based on primary metric and perhaps other metrics
        • TODO Identify best combination
        Find weaknesses
        • TODO.  Try to identify areas where models are performing poorly.  Need a framework for this
        Predict based on training data
        • TODO Use same averaging/ensemble methods in best combination
        • Submit
        Iterate
    """

    def __init__(self, model_list = None, data = None):
        if model_list is None:
            self.model_list = []
        else:
            self.model_list = model_list
        self.model_names = []
        self.df_predictions_train = None
        self.df_predictions_test = None
        self.df_predictproba_train = None
        self.df_predictproba_test = None
        self.num_classes = None 
        self.__metric = None
        self.is_fully_trained = False
        if data is not None:
            self.load_dataset(data)


    def load_dataset(self, data):
        assert isinstance(data, dc_dataset), f"{data} should be an instance of dc_dataset"
        self.data = data
        self.num_classes = self.data.guess_num_classes()


    def get_input_shape(self):
        assert self.data.x_train is not None, "Need to process data first"
        return self.data.x_train.shape[1]


    # TODO, make num_observations work for 
    def model_search(self, num_observations = math.inf, use_kfold = False):
        """
        This performs a rough search across common models to identify and shortlist candidate models
        """
        self.model_list = [] # clear existing model list
        self.model_names = []
        if self.num_classes == 0:
            self.add_models([
                #build_Keras_regressor([64,64,64], 0.5, self.get_input_shape() ),
                XGBRegressor(),
                RandomForestRegressor(max_depth=7, n_estimators=100, max_features=5)
            ])

        elif self.num_classes == 1:
            self.add_models(
                [
                build_Keras_binary_classifier([64,64,64], 0.5, self.get_input_shape() ),
                XGBClassifier(),
                KNeighborsClassifier(3),
                SVC(kernel="linear", C=0.025, probability=True),
                SVC(gamma=2, C=1, probability=True),
                GaussianProcessClassifier(1.0 * RBF(1.0)),
                DecisionTreeClassifier(max_depth=7),
                RandomForestClassifier(max_depth=7, n_estimators=20, max_features=1),
                MLPClassifier(alpha=1),
                AdaBoostClassifier()
            ],[ 
                "KerasNN", "XGB", "Nearest Neighbors", "Linear SVM", "RBF SVM", 
                "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]
            )
        if use_kfold:
            print(self.assess_kfold() )
        else:
            print(self.assess_holdback() )


    def add_models(self, model_list, names = None):
        """ 
        Add models to the project.  Can pass a model or a list of models.  
        Models can be a Sklearn, Keras, XGBoost or LGBM model or a dc_model instance (wrapper for above).
        """
        # convert to a list if only one is passed
        if not isinstance(model_list, (list, tuple)):
            model_list = [model_list]
        if names is None:
            names = [None] * len(model_list)
        if not isinstance(names, (list, tuple)):
            names = [names]
        assert len(model_list) == len(names), \
               f"nodel_list has length {len(model_list)}\nnames has length {len(names)}\n "
        for m, n in zip(model_list, names):
            if isinstance(m, dc_model):
                self.model_list.append(m)
            else:
                self.model_list.append(dc_model(m, self.num_classes, n))
            # OLD: assert isinstance(m, dc_model), f"{n} should be an instance of dc_model"
            self.model_names.append(m.__str__() ) # use m._str__ rather than n in case name has been created by dc_model constructor


    def train_final(self):
        for model in self.model_list:
            print (f"Training model {str(model)}")
            model.fit(self.data.x_train, self.data.y_train)


    def predict(self):
        assert all([m.is_fully_trained == True for m in self.model_list]), "Not all models are trained."
        self.df_predictions_train = pd.DataFrame(
            [self.data.y_train], columns="Actual")
        self.df_predictions_test = pd.DataFrame(
            [self.data.x_test.iloc[:,0]], columns="TEMP") # add first col of x as placeholder to get correct index and sizing
        for model,n in zip(self.model_list, self.model_names):
            self.df_predictions_train[n] = model.predict(self.data.x_train)
            self.df_predictions_test[n]  = model.predict(self.data.x_test )
        self.df_predictions_test.drop(columns="TEMP") # remove placeholder


    def predict_proba(self):
        """
        Percentage (i.e. 0 to 1) probability prediction for each class
        """
        assert self.num_classes > 0 , \
               "predict_proba() not relevant for regressor.  Use predict() instead?"
        self.df_predictproba_train = pd.DataFrame(
            [self.data.y_train], columns="Actual")
        self.df_predictproba_test = pd.DataFrame(
            [self.data.x_test.iloc[:,0]], columns="TEMP") # add first col of x as placeholder to get correct index and sizing
        for model,n in zip(self.model_list, self.model_names):
            self.df_predictproba_train[n] = model.predict_proba(self.data.x_train)
            self.df_predictproba_test[n]  = model.predict_proba(self.data.x_test )
        self.df_predictproba_test.drop(columns="TEMP") # remove placeholder


    def set_metric(self, metric):
        """
        Set the default metric to assess for this project - e.g. accuracy or RMSE.
        Should set manually, but if not set, dc_model will guess based on type of problem
        """
        self.__metric = metric


    def get_metric(self):
        if self.__metric is None:
            if self.num_classes == 0:
                self.__metric = metrics.mean_squared_error
                print ("WARNING - no metric set.  Using mean squared error.")
                return metrics.mean_squared_error 
            else:
                self.__metric = metrics.accuracy_score
                print ("WARNING - no metric set.  Using accuracy.")
                return metrics.accuracy_score
        else:
            return self.__metric


    def __assess_metric(self, x_data, y_data):
        self.__metric(x_data, y_data)


    def assess_holdback (self, size = 0.2):
        """
        Returns the scores for each model across the training and validation sets based on a simple holdback model
        """
        multimodel_scores = defaultdict(list)

        if self.data.x_val is None:
            self.data.set_validation_data(size)

        for model, name in zip(self.model_list, self.model_names):
            model.fit(self.data.x_partial_train, self.data.y_partial_train)
        
            multimodel_scores[name] = [ 
                self.get_metric()(
                    self.data.y_partial_train, 
                    model.predict(self.data.x_partial_train)),
                self.get_metric()(
                    self.data.y_val, 
                    model.predict(self.data.x_val))
            ]
        # Note: doesn't actually need mean (as only one pass), but copied from kfold                        
        print("\t\t\tTraining\tValidation")    
        for i in multimodel_scores:
            print (f"{i[:20]}:\t{np.mean(multimodel_scores[i][0])}\t{np.mean(multimodel_scores[i][1])}")

            
    def assess_kfold (self, folds = 4, max_observation_proportion = 1.0, verbose = 1):
        """
        Returns the scores for each model across the training and validation sets based on kfold cross validation
        """
        assert max_observation_proportion >= 0 and max_observation_proportion <= 1, \
               f"max_observation_proportion = {max_observation_proportion}\nNeeds to be between 0 and 1"
        max_observations = int(self.data.x_train.shape[0] * max_observation_proportion)
        X = self.data.x_train[:max_observations]
        Y = self.data.y_train[:max_observations]   

        num_val_samples = len(self.data.x_train) // folds
        short_names_for_df = [n[:25] for n in self.model_names]
        df = pd.DataFrame(index = short_names_for_df, columns=["Train", "Val"])
        multimodel_scores = defaultdict(list)
        
        for i in range(folds):
            print(f'\nProcessing fold {i}: ', end='\t')
            val_data = X[i * num_val_samples: (i+1) * num_val_samples].values
            val_targets = Y[i * num_val_samples: (i+1) * num_val_samples].values

            partial_X_train = np.concatenate(
            [X[:i * num_val_samples], X[(i+1) * num_val_samples:]], axis=0)

            partial_y_train = np.concatenate(
            [Y[:i * num_val_samples], Y[(i+1) * num_val_samples:]], axis=0)
            
            # Need to instantiate fresh untrained copies of each model prior to training on fold
            temp_models = [m.get_deep_copy() for m in self.model_list]
 
            for index, model in enumerate(temp_models):
                model.fit(partial_X_train, partial_y_train)
                #multimodel_scores[index].append( self.get_metric()(val_targets, model.predict(val_data)) )
                multimodel_scores[index].append([ 
                    self.get_metric()(partial_y_train, model.predict(partial_X_train)),
                    self.get_metric()(val_targets, model.predict(val_data))
                ])
                if verbose > 0:
                    print (f"Model {index}     ", end='')

        for i,mm in enumerate(multimodel_scores):
            df.iloc[i,0] = np.mean(multimodel_scores[i][0])
            df.iloc[i,1] = np.mean(multimodel_scores[i][1])
        
        return df


    # Question if this is actually meaningful if final should be trained on all training data
    # def assess_final(self):
    #     assert self.df_predictions_train is not None, "Need to call predict() first"
    #     for c in self.df_predictions_train.columns[1:]:
    #         self.__assess_metric(self.df_predictions_train["Actual"], self.df_predictions_train[c])









    # #TODO
    # # Graph this with heatmap?
    # def feature_importances_by_model(self):
    #     fi_list = []
    #     for model in self.model_list:
    #         fi_list.append[] 
    #         pass #TODO

    #     df_feature_impt = pd.DataFrame([self.data.x_train.columns, fi_list], index = ["Feature","Importance"]).transpose()
    #     df_feature_impt.set_index("Feature")
    #     # TODO Question does this actually change the importances to abs values, losing data on whether positively or negatively correlated
    #     df_feature_impt.reindex(df_feature_impt.Importance.abs().sort_values(ascending=False).index)
    #     return df_feature_impt


    # def feature_correlation(self):
    #     # TODO copy grid model from Titanic
    #     pass




class dc_helpers:

    @staticmethod
    def move_col_right(df, col_name):
        temp = df[col_name]
        df.drop(labels=[col_name], axis=1, inplace = True)
        df[col_name] = temp


    # Produces a table summarizing the properties of the features (similar to df.info() )
    @staticmethod
    def get_feature_info(df):
        df1 = pd.DataFrame(df.columns)
        df1["DataType"] = [df[c].dtype for c in df.columns]
        df1 = df1.set_index(0)
        df1["Unique(exclNan)"] = [(len(df[c].unique()) - (
            df[c].isna().any()*1) ) for c in df.columns] 
        df1["DataType"] = [df[c].dtype for c in df.columns]
        df1["Missing"] = [df[c].isna().sum() for c in df.columns]
        df1["Numeric"] = [np.issubdtype(df[c].dtype, np.number) for c in df.columns]
        #df1["Skew"] = ["" if df[c].dtype != np.float64 else df[c].skew() for c in df.columns]
        #df1["LogSkew"] = ["" if df[c].dtype != np.float64 else np.log(df[c]+0.0001).skew() for c in df.columns]
        # for r in df1.index:
        #     if df1
        # return df1
    # dtype, nan, unique vals, max, min, mean, outlier_low, outlier_high, guess[cat,num,binary]


    @staticmethod
    def list_columns(df):
        for i,c in enumerate(df.columns):
            print (i,"\t",c)
        
    











                


def build_Keras_binary_classifier(neuronList, dropout, input_shape):
    model = models.Sequential()
    # build a dropout list if it is passed as a single value
    if not isinstance(dropout, (list, tuple)):
        dropout = [dropout for i in range(len(neuronList))]
    model.add(layers.Dense(neuronList[0], activation="relu", input_shape = (input_shape,) ))
    model.add(layers.Dropout(dropout[0]))
    
    for n,d in zip(neuronList[1:],dropout[1:]):
        model.add(layers.Dense(n, activation="relu"))
        model.add(layers.Dropout(d))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="nadam", loss = "binary_crossentropy", metrics = ["accuracy"])

    return model

def build_Keras_regressor(neuronList, dropout, input_shape):
    #  dropout can be a list e.g. [0.5,0.4,0.3] or a single value
    model = models.Sequential()
    if not isinstance(dropout, (list, tuple)):
        dropout = [dropout for i in range(len(neuronList))]
    model.add(layers.Dense(neuronList[0], activation="relu", input_shape = (input_shape,) ))
    model.add(layers.Dropout(dropout[0]))
    
    for n,d in zip(neuronList[1:],dropout[1:]):
        model.add(layers.Dense(n, activation="relu"))
        model.add(layers.Dropout(d))
    model.add(layers.Dense(1))

    model.compile(optimizer="nadam", loss = "mse", metrics = ["mae"])

    return model


# # wrap the Keras NN so it can use the same [fit, accuracy_score] interface as the sklearn models
# class KerasNN:

#     # PLaceholder for abstract base class    
#     #def __init__(self):
#     #    raise NotImplementedError( "Derived class should implement this" )
    
#     def __init__(self, neuronList, dropout, epochs = 80):
#         self.model = build_Keras_binary_classifier(neuronList, dropout, input_shape = 999)
#         self.name = "KerasNN" + str(neuronList) + str(dropout)
#         self.epochs = epochs
#         self.__hist = None
        
#     def __str__(self):
#         return self.name
        
#     def fit(self, x, y):
#         self.__hist = self.model.fit(x,y,epochs=self.epochs, batch_size=128, verbose = 0)
    
#     #  This assumes binary classifier.  Need to generalize
#     def predict(self, x):
#         return np.round(self.model.predict(x)).astype(int) 
    
#     #  This assumes binary classifier.  Need to generalize
#     def predict_proba(self, vals):
#         return [[float(1-x), float(x)] for x in list(self.model.predict(vals))]
    
    
#     # This is for classifier.  Need to generalize
#     def plot_hist(self, title):
#         if self.__hist is None:
#             print ("Must call fit() before plot_hist()")
#             return
#         history_dict = self.__hist.history
#         loss_values = history_dict['loss']
#         val_loss_values = history_dict['val_loss']

#         epochs = range(1, len(history_dict['acc']) + 1)

#         plt.figure(figsize=(18, 4.5))

#         plt.subplot(1,2,1)
#         plt.plot(epochs, loss_values, 'g', label='Training loss')           
#         plt.plot(epochs, val_loss_values, 'r', label='Validation loss')      
#         plt.title(title)
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()

#         acc_values = history_dict['acc']
#         val_acc_values = history_dict['val_acc']

#         plt.subplot(1,2,2)

#         plt.plot(epochs, acc_values, 'b', label='Training acc')
#         plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.tight_layout()
#         plt.show()


# def get_feature_info(df):
#     df1 = pd.DataFrame(df.columns)
#     df1["DataType"] = [df[c].dtype for c in df.columns]
#     df1 = df1.set_index(0)
#     df1["Unique(exclNan)"] = [(len(df[c].unique()) - (df[c].isna().any()*1) ) for c in df.columns] #- df[c].isna().any()
#     df1["DataType"] = [df[c].dtype for c in df.columns]
#     df1["Missing"] = [df[c].isna().sum() for c in df.columns]
#     df1["Numeric"] = [np.issubdtype(df[c].dtype, np.number) for c in df.columns]
#     df1["Skew"] = ["" if df[c].dtype != np.float64 else df[c].skew() for c in df.columns]
#     df1["LogSkew"] = ["" if df[c].dtype != np.float64 else np.log(df[c]+0.0001).skew() for c in df.columns]
#     return df1




# Test implementation of library with NYC taxi data.  No feature engineering done.
if False:

    pd.options.display.float_format = '{:.04f}'.format

    df = pd.read_csv(r"C:\Users\david\OneDrive\ML 2018\Data\NYC\NYCmed.csv", parse_dates=['pickup_datetime'])
    df_test = pd.read_csv(r"C:\Users\david\OneDrive\ML 2018\Data\NYC\test.csv", parse_dates=['pickup_datetime'])

    df.drop("Unnamed: 0", axis = 1, inplace=True)
    df.drop("key", axis = 1, inplace=True)
    df_test.drop("key", axis = 1, inplace=True)
    temp = df['fare_amount'].astype(np.float32)
    df.drop(labels=['fare_amount'], axis=1, inplace = True)
    df["fare_amount"] = temp

    data = dc_dataset(df, "fare_amount")
    data.load_test(df_test)

    data.raw.describe()

    data.remove_outliers(list(data.raw.columns[1:5]), 0.02, lower_only=True)

    data.remove_outliers(list(data.raw.columns[1:5]), 0.01, upper_only=True)

    data.raw[data.raw["Test"] == 0].describe()

    data.raw.columns

    data.process_raw([],[1,2,3,4,5],[0],[])

    data.processed.describe()

    pr = dc_project()
    pr.load_dataset(data)

    pr.model_search(use_kfold=True)











