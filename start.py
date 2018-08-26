import LocationPrediction.source_data
import LocationPrediction.preprocess_data
import LocationPrediction.predictor
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from  IPython.display import display


def main(argv):
    ############################################################
    # customize pandas print output
    # pd.set_option('display.height', 1000)
    # pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 250)
    ############################################################


    ############################################################
    # data downloading and cleaning
    sd = LocationPrediction.source_data.source_dataClass()
    # download and clean data
    # df_clean_csv = sd.download_and_clean() # takes time because it reads individual source csv files, combines them, cleans them
    # read the cleaned csv into memory
    df_clean_csv = sd.read_clean_csv()
    ############################################################


    ############################################################
    # data preprocessing up to the generation of a training dataframe
    prpdata_train = LocationPrediction.preprocess_data.preprocess_dataClass(df=df_clean_csv)

    # create a reduced dataset by restricting observations to fall into a specific date range
    # this also creates distinct indices
    min_date = '2017-02-01'
    max_date = None
    n_cuts = 5
    N_positive_samples_to_draw = -1
    Neg_Multiplier = 1
    prpdata_train.create_training_data(min_date=min_date,
                                       n_cuts=n_cuts,
                                       N_positive_samples_to_draw=N_positive_samples_to_draw,
                                       Neg_Multiplier=Neg_Multiplier)

    print(prpdata_train)
    print(prpdata_train.df_training_one_hot_vector.head())
    ############################################################


    ############################################################
    # # create a random forest and train it with the training data
    # rf = RandomForestClassifier(n_estimators=2000,
    #                                      random_state=42,
    #                                      class_weight='balanced',
    #                                      n_jobs=-1)
    #
    # df_train = prpdata_train.df_training_one_hot_vector.copy()
    # X_df_train = df_train.drop(columns=[prpdata_train.label_column_name])
    # X_train = X_df_train.values
    #
    # y_df_train = df_train.drop(columns=df_train.columns.difference([prpdata_train.label_column_name]))
    # y_train = y_df_train.values
    #
    # rf.fit(X_train,y_train.ravel())

    filename = 'my_rf.pkl'
    # pickle.dump(rf, open(filename, 'wb'))
    rf = pickle.load(open(filename,'rb'))
    ############################################################


    ############################################################
    # make a prediction using the trained random forest
    # the features of the first row of the training data
    # make a prediction using the trained random forest
    # the features of the first row of the training data

    df_with_known_matches = prpdata_train.df_selection_categorical_discretized_locations.copy()
    df_with_known_matches = df_with_known_matches[prpdata_train.columns_to_keep_categorical].drop_duplicates()
    df_with_known_matches = df_with_known_matches[df_with_known_matches[prpdata_train.label_column_name]==1].drop(columns=[prpdata_train.label_column_name])
    display(df_with_known_matches.head(n=10))

    predictor = LocationPrediction.predictor.predictorClass(model=rf,
                                                            preprocessed_data_for_training=prpdata_train,
                                                            df_with_known_matches=df_with_known_matches)


    my_dict = {'price': 100,
               'room_type':'Private room',
               'bedrooms': 2,
               'accommodates': 2}
    predictor.set_features_for_prediction(int_or_dict=my_dict)
    predictor.set_locations_for_prediction(df_with_locations=None)

    # create the cross join of features and locations to generate a dataframe for prediction
    predictor.set_features_and_locations()
    display(predictor.df_features_and_locations_to_predict)
    display(predictor.df_features_and_locations_one_hot_vector_to_predict)



    y_probas =  predictor.predict_probas(beta=prpdata_train.beta_for_training_data)
    # y_probas = predictor.predict(beta=1)
    predictor.fill_prediction_dataframe_with_probas(y_probas=y_probas)

    df_predictions = predictor.df_features_and_locations_to_predict

    display(df_predictions.head(n=10))
    display(df_predictions.sort_values(by='probability',ascending=False).head(n=10))



    ############################################################


    ############################################################
    # plot the prediction
    ############################################################


    ############################################################
    # evaluate the fit
    ############################################################


    pass

    # # create the source data directory and delete anything that was previously there
    # source_data_dir = os.path.join(work_dir,'source_data')
    # LocationPrediction.source_data.create_dir(data_dir=source_data_dir, overwrite=True)
    #
    # # download the zipped sourcefiles from the internet

if __name__ == '__main__':
    sys.exit(main(sys.argv))
    pass