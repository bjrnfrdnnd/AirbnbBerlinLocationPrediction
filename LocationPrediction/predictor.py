from sklearn.ensemble import RandomForestClassifier
from LocationPrediction.preprocess_data import  preprocess_dataClass
from LocationPrediction.preprocess_data import  fix_columns
import pandas as pd
import numpy as np

class predictorClass(object):
    def __init__(self,
                 model:RandomForestClassifier=None,
                 preprocessed_data_for_training:preprocess_dataClass=None,
                 df_with_known_matches:pd.DataFrame=None):

        assert (isinstance(model, RandomForestClassifier))
        assert (isinstance(preprocessed_data_for_training, preprocess_dataClass))
        assert (isinstance(df_with_known_matches, pd.DataFrame))


        self.model = model
        self.preprocessed_data_for_training = preprocessed_data_for_training
        self.df_features = None
        self.df_locations = None
        self.df_features_and_locations_to_predict = None
        self.df_features_and_locations_one_hot_vector_to_predict = None
        self.df_with_known_matches = None
        self.prediction_method = None

        self.y_proba_tiny = 1.e-20 # for some reason, setting the proba to zero shows up like a 1 in the heatmap

        self.set_df_with_known_matches(df_with_known_matches=df_with_known_matches)

    def set_df_with_known_matches(self,
                                  df_with_known_matches:pd.DataFrame=None):
        assert (isinstance(df_with_known_matches, pd.DataFrame))
        self.df_with_known_matches = df_with_known_matches


    def set_features_and_locations(self,
                                   df_features: pd.DataFrame = None,
                                   df_locations: pd.DataFrame = None):
        """combines features and locations to one dataframe using cross join

        If any of the arguments is None, it is replaced by the corresponding members
        The one_hot_vector version for predictions with the random forest is also created (member df_features_and_locations_one_hot_vector_to_predict)

        :param df_features: a dataframe with features or None (one row)
        :type df_features: pd.DataFrame
        :param df_locations: a dataframe with locations or None (multiple rows)
        :type df_locations: pd.DataFrame
        :return:
        :rtype:
        """

        if df_features is not None:
            self.df_features = df_features
        if df_locations is not None:
            self.df_locations = df_locations

        # cross join
        self.df_features_and_locations_to_predict = pd.merge(self.df_features.assign(key_to_join=0),
                       self.df_locations.assign(key_to_join=0),
                       on='key_to_join').\
            drop('key_to_join',
                 axis=1)

        # drop duplicates and reset index
        self.df_features_and_locations_to_predict = self.df_features_and_locations_to_predict.drop_duplicates().reset_index(drop=True)

        # create one_hot_vector version
        self.df_features_and_locations_one_hot_vector_to_predict = \
            self.preprocessed_data_for_training.apply_one_hot_vector_encoding(
                df_categorical=self.df_features_and_locations_to_predict)

        # fix columns of one_hot_vector version
        column_names_in_training_data = \
            self.preprocessed_data_for_training.df_training_one_hot_vector.columns.drop(
            self.preprocessed_data_for_training.label_column_name,
            errors='ignore')

        self.df_features_and_locations_one_hot_vector_to_predict = \
            fix_columns(
                df=self.df_features_and_locations_one_hot_vector_to_predict,
                column_names=column_names_in_training_data)


        pass

    def predict_probas(self, beta:float=1):
        """use the members 'model' and 'self.df_features_and_locations_one_hot_vector_to_predict' to predict probabilities

        * adjust the probabilities by beta

        :param beta: probability to sample a negative sample from the entire selection in the training set
        :type beta: float

        :return: list of probabilities
        :rtype: np.ndarray
        """

        # find out which of the rows in the in the dataframe with known matches correspons to requested combination of features and locations
        # the boolean could be a list of only 'False' values

        boolean_indicator_of_matches = pd.merge(self.df_features_and_locations_to_predict,
                                                self.df_with_known_matches,
                                                how='left', indicator=True)['_merge'].isin(['both'])
        if sum(boolean_indicator_of_matches)>0:
            # we found matches that already exist
            print('found existing matches: {}'.format(sum(boolean_indicator_of_matches)))
            # set the probabilities of the corresponding rows to 1 and the rest to 0
            y_probas = np.ones(self.df_features_and_locations_to_predict.shape[0]) * self.y_proba_tiny
            y_probas[boolean_indicator_of_matches] = 1.0
            self.prediction_method = 'lookup'

        else:
            # no matches found. We proceed using the random forest to predict
            X_features = self.df_features_and_locations_one_hot_vector_to_predict.values
            y_probas = self.model.predict_proba(X_features)
            y_probas = y_probas[:,1]

            # adjust probas using beta
            y_probas = beta * y_probas / (y_probas * (beta - 1) + 1)
            y_probas[y_probas==0] = self.y_proba_tiny
            self.prediction_method = 'prediction'

        return y_probas

    def fill_prediction_dataframe_with_probas(self, y_probas):
        """use y_probas to fill the member df_features_and_locations_to_predict; add information of how the prediction was done


        :param y_probas: probability that a given combination of features and locations is contained in the Airbnb listings
                 This is a list with as many entries as the member df_features_and_locations_to_predict has rows
        :type y_probas: np.ndarray

        :return: None
        :rtype: None
        """

        df_predictions = self.df_features_and_locations_to_predict[
            self.preprocessed_data_for_training.feature_column_names + \
            self.preprocessed_data_for_training.location_column_names
        ].copy()
        df_predictions['probability'] = y_probas
        df_predictions['class'] = df_predictions['probability'] > 0.5
        df_predictions['prediction_method'] = self.prediction_method

        self.df_features_and_locations_to_predict = df_predictions.copy()

    def set_features_for_prediction(self,
                                    int_or_dict=None):
        """ set the features we sich to predict

        * fills the member df_features with a one-row dataframe corresponding to the input

        :param int_or_dict: integer into the rows of the training dataset or a dictionary that specifies the features
        :type int_or_dict:
        :return: None
        :rtype: None
        """


        # prepare a dataframe that contains the most frequent values of each column
        df_mode = self.preprocessed_data_for_training.df_selection_categorical_discretized_locations[
            self.preprocessed_data_for_training.feature_column_names].mode()

        df_features_for_prediction = df_mode.copy()

        if isinstance(int_or_dict,dict):
            # we passed a dictionary with elements that are to be used for the features
            for colname in self.preprocessed_data_for_training.feature_column_names:
                v = int_or_dict.get(colname,None)
                if v is not None:
                    df_features_for_prediction.loc[0,colname] = v
                    pass
                pass
            pass
        else:
            # we passed a row number that indexes a row of the training dataframe
            df_features_for_prediction = self.preprocessed_data_for_training.\
                df_selection_categorical_discretized_locations.loc[
                int_or_dict:int_or_dict,
                self.preprocessed_data_for_training.feature_column_names]
            pass


        self.df_features = df_features_for_prediction

    def set_locations_for_prediction(self,
                                    df_with_locations:pd.DataFrame=None):
        """ set the locations that are to be combined with the features for prediction

        if df_with_locations is None, then choose all the distinct locations from the training dataset
        else, use the dataframe to fill the locations

        * fills the member df_locations with rows corresponding to the input

        :param df_with_locations: None or a dataframe with locations
        :type df_with_locations: pd.DataFrame
        :return: None
        :rtype: None
        """


        # by default, fill the dataframe with the contents of the training dataset
        df_locations_for_prediction = self.preprocessed_data_for_training.df_selection_categorical_discretized_locations[self.preprocessed_data_for_training.location_column_names]
        df_locations_for_prediction = df_locations_for_prediction.drop_duplicates().reset_index(drop=True)


        if isinstance(df_with_locations,pd.DataFrame):
            df_new = None
            try:
                df_new = df_with_locations[self.preprocessed_data_for_training.location_column_names]
            except:
                pass
            pass
            if df_new is not None:
                df_locations_for_prediction = df_new
                pass
            pass

        self.df_locations = df_locations_for_prediction


def initialize(self):
        pass
