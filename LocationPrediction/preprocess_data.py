import pandas as pd
import re
from typing import List
import numpy as np
import collections

def add_missing_dummy_columns( df:pd.DataFrame=None,
                               column_names:collections.Iterable=None):
    """add columns to df that are in column_names but not in df.columns

    Used for one_hot_vector encoding of datasets containing less category levels
    than the training data
    * changes df inplace

    :param df: the dataframe to add columns to
    :type df: pd.DataFrame
    :param column_names: the list of columns that needs to be present in the dataframe
    :type column_names: L
    :return:
    :rtype:
    """
    assert(isinstance(df,pd.DataFrame))
    assert(isinstance(column_names, collections.Iterable))

    missing_cols = set( column_names ) - set( df.columns )
    for c in missing_cols:
        df[c] = 0


def fix_columns(df:pd.DataFrame=None,
                column_names:collections.Iterable=None):
    """fix columns for one_hot_vector_encoding

    * This function adds those columns to df that are in column_names but not in df_columns
    * it sets those columns to zero
    * it removes columns from df if they are not in column_names
    * it reorders the columns of df such that they match the order of the columns in column_names

    :param df: dataframe to be adjusted
    :type df: pd.DAtaFrame
    :param column_names: the list of columns that should exist in df
    :type column_names: collections.Iterable
    :return: dataframe with adjusted columns
    :rtype: pd.DataFrame
    """

    assert(isinstance(df,pd.DataFrame))
    assert(isinstance(column_names, collections.Iterable))

    add_missing_dummy_columns(df=df,
                              column_names=column_names)

    # make sure we have all the columns we need
    assert(set(column_names) - set(df.columns) == set())

    extra_cols = set(df.columns) - set(column_names)
    if extra_cols:
        print ("extra columns:", extra_cols)

    df = df[column_names]
    return df

def calc_N_negative_samples(
                            N_positive_samples_to_draw:int,
                            N_available_positive_samples:int,
                            N_possible_combinations:int,
                            Neg_Multiplier: int = 1,
                            ):
    """calculate the number of negative samples to draw

    If N_positive_samples_to_draw == -1, we draw all available positive samples

    If Neg_Multiplier > 0, we want to draw Neg_Multiplier*N_positive_samples_to_draw negative samples
    If Neg_Multiplier == -1, we want to draw negative samples such that the resulting dataset has the
       same ratio between negative samples and positive samples as the selection from which the samples are drawn

    :param N_positive_samples_to_draw: The number of positive samples we wish to draw. Can not be larger than N_available_positive_samples
    :type N_available_positive_samples: int
    :param N_available_positive_samples: The number of available positive samples in the selection
    :type N_available_positive_samples: int
    :param N_possible_combinations: The number of possible combinations from which to draw negative samples
        (N_possible_negatives = N_possible_combinations - N_available_positive_samples)
    :type N_possible_combinations: int
    :param Neg_Multiplier: The multiplicity of negative samples to draw with respect to positive samples to draw
    :type Neg_Multiplier:  float

    :return:
    :rtype:
    """

    N_possible_negatives = N_possible_combinations - N_available_positive_samples
    if N_positive_samples_to_draw == -1:
        N_positive_samples_to_draw = N_available_positive_samples
    N_positive_samples_to_draw = min(N_positive_samples_to_draw, N_available_positive_samples)

    if Neg_Multiplier == -1:
        # draw negative samples such that the ratio of negative to positive samples is the same as in the selection
        N_negative_samples_to_draw = int(N_positive_samples_to_draw * N_possible_negatives / N_available_positive_samples)
    else:
        N_negative_samples_to_draw = N_positive_samples_to_draw * Neg_Multiplier
        N_negative_samples_to_draw = min(N_possible_negatives, N_negative_samples_to_draw)
        Neg_Multiplier = N_negative_samples_to_draw / N_positive_samples_to_draw

    return N_negative_samples_to_draw

def map_integers_to_larger_interval(N_total=12,
                                    integers_to_exclude=np.array([1, 2, 4, 8, 10]),
                                    integers_to_map=np.arange(7)
                                    ):
    """map a list of integers to a larger interval while not allowing to map to integers within integers_to_exclude

    map a list of integers ('integers_to_map') that are in [0,N_total-len(integers_to_exclude)) to the interval
    [0, N_total) such that no integer in the list 'integers_to_exclude' (integers in [0,N_total)) is being mapped to.

    :param N_total: we map to an interval [0,N_total). integers_to_exclude are within that interval
    :type N_total: int
    :param integers_to_exclude: the list of integers to exclude. They are within [0, N_total)
    :type integers_to_exclude: List[int] or nparray of ints
    :param integers_to_map: list of integers in [0,N_total-len(integers_to_exclude)) to be mapped to [0,N_total)
    :type integers_to_map: List[int] or nparray of ints
    :return:
    :rtype:
    """

    if not isinstance(integers_to_exclude,np.ndarray):
        integers_to_exclude = np.array(integers_to_exclude)

    if not isinstance(integers_to_map,np.ndarray):
        integers_to_map = np.array(integers_to_map)


    high_limit_integers_to_map = N_total - len(integers_to_exclude)
    assert(high_limit_integers_to_map>0)
    assert(len(integers_to_exclude)>0)
    assert(len(integers_to_map)>0)

    assert(min(integers_to_exclude)>=0)
    assert (max(integers_to_exclude) < N_total)
    assert (len(integers_to_exclude) == len(np.unique(integers_to_exclude)))

    assert (min(integers_to_map) >= 0)
    assert (max(integers_to_map) < high_limit_integers_to_map)
    assert (len(integers_to_map) == len(np.unique(integers_to_map)))



    integers_to_exclude_sorted = np.sort(integers_to_exclude)

    # prepare a new list of integers to exclude by prepending -1
    # this is done to assure that the list of gaps generated below also contains a correct value
    # for the first entry
    integers_to_exclude_prime = np.concatenate([[-1], integers_to_exclude_sorted])
    # print('integers to exclude prime', integers_to_exclude_prime)

    # the integers between subsequent integers to exclude is defined as 'gap'
    # These 'gaps' contain  the integer candidates ('free' integers) to which we want to map.
    # The first 'gap' is the number of 'free' integers before the first integer to exclude
    # The index at which each 'gap' starts is [integers_to_exclude] + 1
    # Because we prepended [integers_to_exclude] with -1, the starting index of the first gap is correctly calculated
    indices_at_beginning_of_gaps = integers_to_exclude_prime + 1
    # print('indices at beginning of gaps', indices_at_beginning_of_gaps)

    # each gap is characterized by a length: the number of 'free' integers that can be put into the gap
    # the length of the gap is therefore the diff of integers_to_exclude -1
    # the length of the last gap is the number of integers between the highest possible index (index_max = N_total - 1)
    # and the last integer to exclude
    index_max = N_total -1
    last_index_to_exclude = integers_to_exclude_prime[-1]
    lengths_of_gaps = np.concatenate([
        np.diff(integers_to_exclude_prime) - 1,
        [index_max - last_index_to_exclude]])
    # print('lengths of gaps', lengths_of_gaps)

    # the aim is to find the correct gap for integers_to_map.
    # this can be done by binning the integers_to_map into bins where the limits are given by the cumulative
    # sum of the lengths_of_gaps
    # the upper bound of bin i is given by bin_limit_i
    # the lower bound of bin i is given by bin_limit_i_minus_1
    # a number is sorted into bin_i if bin_limit_i_minus_1 <= number < bin_limit_i
    # all numbers are sorted in one of the bins (from 0 to n_bins-1). The lower bound of the lowest bin is minus infinity
    # we define bin_limit_minus_one[0] as 0.
    # this will not affect the binning itself as only bin_limit_i is used.
    # however, the lowest number to sort is 0 (all intervals considered start at 0)
    # for the calculation of the correct mapping, we will use bin_limit_i_minus_1
    # which necessitates that bin_limit_i_minus_1[0] is 0
    bin_limit_i = np.cumsum(lengths_of_gaps)
    bin_limit_i_minus_one = np.roll(bin_limit_i, 1)
    bin_limit_i_minus_one[0] = 0
    # print('bin_limit_i', bin_limit_i)
    # print('bin_limit_i_minus_one', bin_limit_i_minus_one)

    # sort the integers_to_map into the bins
    bin_indices = np.digitize(integers_to_map, bin_limit_i)
    # print('bin_indices', bin_indices)

    # we can now calculate the integers in [0, N_total) to which we mapped
    integers_mapped = indices_at_beginning_of_gaps[bin_indices] + \
                          integers_to_map - \
                          bin_limit_i_minus_one[bin_indices]
    # print('integers_mapped', integers_mapped)

    # check that no mapped index is contained in the integers_to_exclude
    a = set(integers_to_exclude)
    b = set(integers_mapped)
    c = a.intersection(b)
    assert(len(c)==0)

    # check that all mapped indices are in [0,N_total)
    assert(max(integers_mapped)<N_total)
    assert (min(integers_mapped) >= 0)

    return (integers_mapped)


def get_distinct_integers_without_replacement(N_high,
                                              size=100,
                                              method='direct',
                                              random_state=42):
    """draw distinct integers from [0,N_high) with a uniform distribution without replacement

    We are faced with the problem of having to draw integers in [0,N_high) without replacement,
    where N_high can be potentially large (in the billions or more due to the fact that N_high corresponds
    to N_possible_negatives which can be several billions.
    In this case, drawing samples without replacement is a challenge as typically we would have to build a set
    consisting of all possible integers and then drawing size samples from that set without replacement.
    The choice function of numpy and pandas operates that way if we pace replace=False.
    This can lead to out-of-memory errors when N_high is too large.
    We call this first method the 'direct' method, which is ok if N_high is not too large (several millions)

    When N_high is larger, we apply a different method that we call 'indirect'. Instead of drawing samples without
    replacement, we draw size samples with replacement (which costs no memory as no set of size N_high is built).
    We then check if the drawing resulted into at least size distinct samples. If that is not the case, we repeat
    drawing size samples and add these samples to a list of distinct results from prior drawings.
    We continue this process until we have at least size distinct samples.

    :param N_high: integers are drawn from [0,N_high)
    :type N_high: int
    :param size: the number of integers to be drawn
    :type size: int
    :param method: the method to be used. 'direct' uses the numpy choice function with replace=False, otherwise replace=True repeatedly
    :type method: str
    :param random_state: the random_state to be used. Anything that can be converted to numpy.random.RandomState
    :type random_state: int or numpy.random.RandomState

    :return:
    :rtype:
    """

    # initialize the result value
    result = None

    # conform random_state to be of type numpy.random.RandomState
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        pass
    assert (isinstance(random_state, np.random.RandomState))

    # if size==N_high, we want all integers in [0,N_high)
    if size==N_high:
        a = np.arange(N_high)
    else:
        # draw samples
        if method == 'direct':
            # apply direct method using replace=False
            # an array of size N_high is formed behind the scenes.
            # if N_high is large, this can lead to high memory consumption or out-of-memory errors
            a = random_state.choice(N_high, size=size, replace=False)
        else:
            # continuously draw samples with replacement until a number of size distinct samples are found
            a = random_state.choice(N_high, size=size, replace=True)
            a = np.unique(a)
            # print('first: {}/{}, fraction {}'.format(len(a),size,len(a)/size))
            if len(a) >= size:
                # we found a set of unique samples that is larger or equal to size
                pass
            else:
                while True:
                    b = random_state.choice(N_high, size=size, replace=True)
                    b = np.unique(b)
                    # print('next draw: {}/{}, fraction {}'.format(len(b), size, len(b) / size))
                    a = np.concatenate([a, b])
                    a = np.unique(a)
                    # print('coombined array: {}/{}, fraction {}'.format(len(a), size, len(a) / size))
                    if (len(a) >= size):
                        break
            a = a[:size]
        # print('final',len(a))
    result = a

    return result


class preprocess_dataClass(object):
    def __init__(self, df=None):
        assert(isinstance(df,pd.DataFrame))

        self.df_original_clean = df.copy()
        self.date_column_name = 'last_modified'
        self.feature_column_names = ['price','bedrooms','accommodates','room_type']
        self.location_column_names = ['latitude', 'longitude']
        self.feature_index_column_name = 'feature_index'
        self.flat_index_column_name = 'flat_index'
        self.location_index_column_name = 'location_index'
        self.label_column_name = 'label' # 0 for negative samples, 1 for positive
        self.columns_to_keep_categorical = self.feature_column_names + self.location_column_names + [self.label_column_name]
        # contains the list of columns in the categorical dataset that are to be transformed to one_hot_vector
        self.one_hot_vector_category_column_names = ['room_type']


        self.df_selection_categorical_undiscretized_locations = None
        self.df_selection_categorical_discretized_locations = None


        self.N_selection_distinct_features = 0
        self.N_selection_distinct_locations = 0
        self.N_selection_positives = 0 # total number of rows in the selection from the original dataset
        self.N_selection_distinct_combinations = 0 # number of all possibilities to combine features and locations
        self.N_selection_possible_negatives = 0 # number of possible negatives: N_selection_distinct_combinations - N_selection_positives

        self.N_training_positives = 0
        self.N_training_negatives = 0
        self.N_training_total = 0
        self.N_training_distinct_features = 0
        self.N_training_distinct_locations = 0


        # default settings
        self._method_to_use_for_sampling = 'indirect' # continuously draw samples with replacement until enough distinct samples are found; this is low on memory consumption
        self._random_state = 42
        if not(isinstance(self._random_state,np.random.RandomState)):
            self._random_state = np.random.RandomState(self._random_state)

        # final dataframe after selection, discretization and sampling used for training and validation
        self.df_training_categorical = None
        self.df_training_one_hot_vector = None

        # helper for the calculation of the analytical adjustment of probabilities
        self.beta_for_balanced_data = 0
        self.beta_for_training_data = 0



        pass

    def __repr__(self):
        my_string = ''
        my_string += 'N_selection_positives: {}\n'.format(self.N_selection_positives)
        my_string += 'N_selection_distinct_features: {}\n'.format(self.N_selection_distinct_features)
        my_string += 'N_selection_distinct_locations: {}\n'.format(self.N_selection_distinct_locations)
        my_string += 'N_selection_distinct_combinations: {}\n'.format(self.N_selection_distinct_combinations)
        my_string += 'N_selection_possible_negatives: {}\n'.format(self.N_selection_possible_negatives)
        my_string += 'N_training positives: {}\n'.format(self.N_training_positives)
        my_string += 'N_training negatives: {}\n'.format(self.N_training_negatives)
        my_string += 'N_training total: {}\n'.format(self.N_training_total)
        my_string += 'N_training distinct features: {}\n'.format(self.N_training_distinct_features)
        my_string += 'N_training distinct locations: {}\n'.format(self.N_training_distinct_locations)
        my_string += 'beta_for_balanced_data: {}\n'.format(self.beta_for_balanced_data)
        my_string += 'beta_for_training_data: {}\n'.format(self.beta_for_training_data)

        return my_string



    def apply_one_hot_vector_encoding(self,
                                      df_categorical:pd.DataFrame=None):
        """applies one hot vector encoding to selected columns on the dataframe df_categorical

        :param df_categorical: the dataframe on which to perform the one_hot_vector_encoding
        :type df_categorical: pd.DataFrame

        :return: a dataframe that is one_hot_vector encoded
        :rtype: pd.DataFrame
        """

        assert(isinstance(df_categorical, pd.DataFrame))
        # apply one hot vector enncoding only to colnames in this list
        colnames = self.one_hot_vector_category_column_names
        df_one_hot_vector = pd.get_dummies(df_categorical, columns=colnames)


        return df_one_hot_vector

    def reverse_one_hot_vector_encoding(self,
                                        df_one_hot_vector:pd.DataFrame=None):

        """reverses the application of the one hot vector operation on the dataframe df_one_hot_vector

        :param df_one_hot_vector: the dataframe on which to perform the reversal
        :type df_one_hot_vector: pd.DataFrame

        :return: a dataframe where the one_hot_vector columns are transformed into one categorical column
        :rtype: pd.DataFrame
        """

        assert(isinstance(df_one_hot_vector, pd.DataFrame))

        colnames = self.one_hot_vector_category_column_names

        df_categorical = None
        for colname in colnames:
            # create a dataframe that only contains the columns that were created by
            # the one_hot vector application
            df_only_one_hot_vector_columns = df_one_hot_vector.filter(regex=(f'^{colname}_')).copy()
            if df_only_one_hot_vector_columns.shape[1] == 0:
                # no column has been found
                break
            # rename the one hot vector columns such that they only contain the values of the original categorical
            # variable
            df_only_one_hot_vector_columns.rename(columns=lambda x: re.sub(f'^{colname}_','',x),inplace=True)


            # create a series that contains the categorical variables
            series_with_categorical_variable = df_only_one_hot_vector_columns.idxmax(axis=1)
            # convert the series to a dataframe
            df_with_categorical_variable = series_with_categorical_variable.to_frame()
            # rename the column name to the original column name
            df_with_categorical_variable.columns=[colname]

            # drop the one hot vector columns from the dataframe
            df_everything_except_one_hot_vector_columns = df_one_hot_vector.filter(regex=(f'^(?!^{colname}_)')).copy()

            # join the two to yield the subpart of the dataframe that is categorical for the colname
            df_categorical_subpart = pd.concat(
                [df_with_categorical_variable, df_everything_except_one_hot_vector_columns], axis=1)

            if df_categorical is None:
                df_categorical = df_categorical_subpart
            else:
                df_categorical = pd.concat([df_categorical, df_categorical_subpart],axis=1)


        return df_categorical


    def set_date_limits(self,
                        min_date=None,
                        max_date=None):
        """choose only rows specified by the date

        * selection operates on the member df_original_clean
        * result of the selection is saved to the member df_selection_categorical_undiscretized_locations and
           df_selection_categorical_discretized_locations

        :param min_date:
        :type min_date:
        :param max_date:
        :type max_date:
        :return:
        :rtype:
        """
        df = self.df_original_clean
        DCN = self.date_column_name
        if min_date is None:
            min_date = df[DCN].min()
        if max_date is None:
            max_date = df[DCN].max()

        chooser = df[DCN] >= min_date
        chooser = chooser & (df[DCN] <= max_date)
        self.df_selection_categorical_discretized_locations = df[chooser]

        # shortcut for the dataframe
        df = self.df_selection_categorical_discretized_locations.copy()

        # drop unneeded columns
        df.drop(columns=df.columns.difference(self.columns_to_keep_categorical), inplace=True)
        df.drop_duplicates(inplace=True)

        # reset index
        df.reset_index(drop=True, inplace=True)

        # add a column for the label
        df[self.label_column_name] = 1

        # set the member dataframes
        self.df_selection_categorical_undiscretized_locations = df.copy()
        self.df_selection_categorical_discretized_locations = df.copy()

        # set feature and location index columns
        self.create_distinct_indices()

        # recalculate numbers
        self.calc_numbers_for_selection()


    def calc_numbers_for_selection(self):
        """calculate relevant numbers

        * operates on the member df_selection_categorical_discretized_locations

        :return:
        :rtype:
        """
        df = self.df_selection_categorical_discretized_locations
        # some numbers
        self.N_selection_positives = df.shape[0]

        # the following is alread done in _create_distinct_indices
        # self.N_selection_distinct_features = df[self.feature_column_names].drop_duplicates().shape[0]
        # self.N_selection_distinct_locations = df[self.location_column_names].drop_duplicates().shape[0]

        self.N_selection_distinct_combinations = self.N_selection_distinct_features * self.N_selection_distinct_locations
        self.N_selection_possible_negatives = self.N_selection_distinct_combinations - self.N_selection_positives

        # preparation for analytical correction of probabilities
        self.beta_for_balanced_data = self.N_selection_positives / self.N_selection_possible_negatives




    def _create_distinct_index(self,
                                columns_to_group:collections.Iterable,
                                index_column_name:str):
        """adds a column with an index into a list of distinct entries corresponding to a group of columns

        * operates on the member df_selection_categorical_discretized_locations

        :param columns_to_group: the group of columns for which only unique rows will be considered
        :type columns_to_group: collections.Iterable
        :param index_column_name: the name of the newly created index column
        :type index_column_name: str
        :return:
        :rtype:
        """

        df = self.df_selection_categorical_discretized_locations

        if index_column_name in df.columns:
            df.drop(columns=[index_column_name], inplace=True)

        df_ = df[columns_to_group].copy()
        df_.drop_duplicates(inplace=True)
        df_.reset_index(inplace=True, drop=True)
        df_[index_column_name] = df_.index
        df = df.merge(df_, on=columns_to_group)

        self.df_selection_categorical_discretized_locations = df

    def create_distinct_indices(self):
        """create distinct indices for features and locations

        * operates on the member df_selection_categorical_discretized_locations
        * adds two new columns that contain indices than index the feature/location

        :return:
        :rtype:
        """

        self._create_distinct_index(columns_to_group=self.feature_column_names,
                                    index_column_name=self.feature_index_column_name)
        self._create_distinct_index(columns_to_group=self.location_column_names,
                                    index_column_name=self.location_index_column_name)

        # calculate flat index
        df = self.df_selection_categorical_discretized_locations.copy()

        selection_feature_index = np.array(df[self.feature_index_column_name])
        selection_location_index = np.array(df[self.location_index_column_name])
        N_distinct_features = len(np.unique(selection_feature_index))
        N_distinct_locations = len(np.unique(selection_location_index))


        flat_index = np.ravel_multi_index(multi_index=[selection_feature_index,
                                                       selection_location_index],
                                          dims=[N_distinct_features,
                                                N_distinct_locations],
                                          order='C')

        self.df_selection_categorical_discretized_locations[self.flat_index_column_name] = flat_index

        self.N_selection_distinct_features = N_distinct_features
        self.N_selection_distinct_locations = N_distinct_locations

    def discretize_locations(self,n_cuts:int=10):
        """discretize the locations

        * operates on the member df_selection_categorical_undiscretized_locations
        * resutls are saved in the member df_selection_categorical_discretized_locations
        * all latitudes are binned into n_cuts equidistant categories
        * all longitudes are binned into n_cuts equidistant categories
        * the columns latitude/longitude are replaced by the mid-value of the bins
        * the distinct location index is recalculated
        * relevant numbers are recalculated
        * the total number of different locations will be at most n_cuts**2 (some categories might not appear in the dataset)

        :param n_cuts: number of categories for both longitude and latitude
        :type n_cuts:
        :return:
        :rtype:
        """


        df = self.df_selection_categorical_undiscretized_locations.copy()

        if n_cuts > 1:
            for col_name in self.location_column_names:
                category_name = '{}_category'.format(col_name)
                mid_name = '{}_category_mid'.format(col_name)
                df[category_name] = pd.cut(df[col_name],
                                            n_cuts,
                                            include_lowest=True)
                b = [a.mid for a in df[category_name]]
                df[mid_name] = b
                df.drop(columns=[category_name], inplace=True)
                df.loc[:, col_name] = df[mid_name]
                df.drop(columns=[col_name], inplace=True)
                df.rename(columns={mid_name: col_name}, inplace=True)


            self.df_selection_categorical_discretized_locations = df

        else:
            # we remove all discretization
            self.df_selection_categorical_discretized_locations = df

        # drop duplicates
        self.df_selection_categorical_discretized_locations.drop_duplicates(inplace=True)

        # reset the indices
        self.create_distinct_indices()

        # recalculate numbers
        self.calc_numbers_for_selection()

    def map_feature_or_location_index_to_columns(self,
                                                 df_with_only_index,
                                                 df_with_data_columns,
                                                 mode:str=None):


        assert(np.isin(mode, ['features','locations']))

        if mode == 'features':
            columns_in_dataframe_with_only_index = [
                self.feature_index_column_name,
                self.label_column_name,
                self.flat_index_column_name,
            ]
            columns_in_dataframe_with_data_columns = \
                self.feature_column_names + \
                [self.feature_index_column_name]
            columns_to_join_on = [
                self.feature_index_column_name
            ]

        if mode == 'locations':
            columns_in_dataframe_with_only_index = [
                self.location_index_column_name,
                self.label_column_name,
                self.flat_index_column_name,
            ]
            columns_in_dataframe_with_data_columns= \
                self.location_column_names + \
                [self.location_index_column_name]
            columns_to_join_on = [
                self.location_index_column_name
            ]


        df_result = pd.merge(
            left=df_with_only_index[columns_in_dataframe_with_only_index],
            right=df_with_data_columns[columns_in_dataframe_with_data_columns],
            on=columns_to_join_on)
        df_result.drop_duplicates(inplace=True)
        df_result.reset_index(drop=True, inplace=True)

        return df_result

    def create_data_with_only_indices(self):
        data_woi = data_with_only_indices(df=self.df_selection_categorical_discretized_locations)
        return data_woi

    def create_training_data(self,
                             min_date:str=None,
                             max_date:str=None,
                             n_cuts:int=10,
                             N_positive_samples_to_draw:int=-1,
                             Neg_Multiplier:int=1):
        """create a training dataset

        If Neg_Multiplier == -1, the final sample will have the same proportion of positive samples as the
        date-selected and location-discretized dataset

        * select a subset of rows according to min_date, max_date
        * discretize locations according to n_cuts
        * choose N_positive_samples_to_draw from the available positive samples in the date-selected and location-discretized dataset
        * choose N_positive_samples*NegMultiplier negative samples
        * create a dataframe with this selection in the member df_training_categorical

        :param min_date:
        :type min_date:
        :param max_date:
        :type max_date:
        :param n_cuts:
        :type n_cuts:
        :param N_positive_samples_to_draw:
        :type N_positive_samples_to_draw:
        :param Neg_Multiplier:
        :type Neg_Multiplier:
        :return:
        :rtype:
        """

        ############################################################
        # set the date limits
        # this also recalculates indices and the other used numbers
        self.set_date_limits(min_date=min_date,
                             max_date=max_date)

        # print(self)
        # df = self.df_selection_categorical_discretized_locations
        # print(df.head())
        # print(df.shape)
        ############################################################

        ############################################################
        # discretize locations
        # this also recalculates indices and the other used numbers
        self.discretize_locations(n_cuts=n_cuts)

        # df = self.df_selection_categorical_discretized_locations
        # print('after discretizationn')
        # print(self)
        # print(df.head())
        # print(df.shape)
        ############################################################


        ############################################################
        # calculate the required number of positive samples
        if N_positive_samples_to_draw==-1:
            N_positive_samples_to_draw = self.N_selection_positives

        N_positive_samples_to_draw = min(N_positive_samples_to_draw,self.N_selection_positives)

        # print('N_positive_samples_to_draw',N_positive_samples_to_draw)
        ############################################################


        ############################################################
        # draw N_positive positive samples
        df_with_positive_samples = self.df_selection_categorical_discretized_locations.sample(
            n=N_positive_samples_to_draw,
            replace=False,
            random_state=self._random_state)
        ############################################################


        ############################################################
        # draw negative samples
        # first, specify the number of negative samples to draw by using Neg_Multiplier
        N_negative_samples_to_draw = calc_N_negative_samples(
            N_positive_samples_to_draw=N_positive_samples_to_draw,
            N_available_positive_samples=self.N_selection_positives,
            N_possible_combinations=self.N_selection_distinct_combinations,
            Neg_Multiplier=Neg_Multiplier)
        # print('N_negative_samples to draw',N_negative_samples_to_draw)

        # second, draw the samples
        negative_samples_drawn = get_distinct_integers_without_replacement(
            N_high=self.N_selection_possible_negatives,
            size=int(N_negative_samples_to_draw),
            method=self._method_to_use_for_sampling,
            random_state=self._random_state,
        )
        # print('length of negatives drawn', len(negative_samples_drawn))

        # map the negative samples drawn to flat index
        flat_index_for_negative_samples = map_integers_to_larger_interval(
            N_total=self.N_selection_distinct_combinations,
            integers_to_exclude=self.df_selection_categorical_discretized_locations[self.flat_index_column_name],
            integers_to_map=negative_samples_drawn)

        # map the flat index to a feature/location index
        selection_feature_index, selection_location_index = \
            np.unravel_index(
                indices=flat_index_for_negative_samples,
                dims=[self.N_selection_distinct_features,
                      self.N_selection_distinct_locations],
                order='C')

        # create a dataframe for the negative samples
        df_negative_samples_only_indices = pd.DataFrame(
            {self.flat_index_column_name: flat_index_for_negative_samples,
             self.feature_index_column_name: selection_feature_index,
             self.location_index_column_name: selection_location_index,
             self.label_column_name: 0
             })

        # print(df_negative_samples_only_indices.head())
        # print(df_negative_samples_only_indices.info())

        # map feature index column to feature columns content
        df_negative_samples_indices_and_feature_columns = self.map_feature_or_location_index_to_columns(
            df_with_only_index=df_negative_samples_only_indices,
            df_with_data_columns=self.df_selection_categorical_discretized_locations,
            mode='features'
        )

        # print(df_negative_samples_indices_and_feature_columns.head())
        # print(df_negative_samples_indices_and_feature_columns.info())

        # map location index column to location columns content
        df_negative_samples_indices_and_location_columns = self.map_feature_or_location_index_to_columns(
            df_with_only_index=df_negative_samples_only_indices,
            df_with_data_columns=self.df_selection_categorical_discretized_locations,
            mode='locations'
        )

        # print(df_negative_samples_indices_and_location_columns.head())
        # print(df_negative_samples_indices_and_location_columns.info())

        df_negative_samples = df_negative_samples_indices_and_feature_columns.merge(df_negative_samples_indices_and_location_columns)

        # print(df_negative_samples.head())
        # print(df_negative_samples.info())
        ############################################################


        ############################################################
        # combine negative and positive samples
        df_positive_and_negative_samples = pd.concat(
            [
                df_with_positive_samples[
                    [self.feature_index_column_name,
                     self.location_index_column_name,
                     self.label_column_name
                     ] + \
                    self.feature_column_names + \
                    self.location_column_names
                    ],
                df_negative_samples
            ],
            sort=False)


        df_positive_and_negative_samples.drop(
            columns=df_positive_and_negative_samples.columns.difference(self.columns_to_keep_categorical), inplace=True)

        assert(df_positive_and_negative_samples.drop_duplicates().shape[0] == df_positive_and_negative_samples.shape[0])
        df_positive_and_negative_samples.reset_index(drop=True, inplace=True)

        # print(df_positive_and_negative_samples.head())
        # print(df_positive_and_negative_samples.info())
        ############################################################


        ############################################################
        # create training dataset and one-hot-vector versions
        self.df_training_categorical = df_positive_and_negative_samples
        ############################################################


        ############################################################
        # calculation of beta for this training dataset
        N_plus_training = self.df_training_categorical[self.df_training_categorical[self.label_column_name] == 1].shape[0]
        N_total_training = self.df_training_categorical.shape[0]
        p_plus_training = N_plus_training/ N_total_training
        self.beta_for_training_data = (1/p_plus_training - 1) * self.N_selection_positives / self.N_selection_possible_negatives
        ############################################################


        ############################################################
        # calculation of numbers for the training (sampled) data
        self.N_training_positives = \
            self.df_training_categorical[self.df_training_categorical[self.label_column_name]==1].shape[0]
        self.N_training_negatives = \
        self.df_training_categorical[self.df_training_categorical[self.label_column_name] == 0].shape[0]
        self.N_training_total = self.N_training_negatives + self.N_training_positives
        self.N_training_distinct_features = \
            self.df_training_categorical[self.feature_column_names].drop_duplicates().shape[0]
        self.N_training_distinct_locations = \
            self.df_training_categorical[self.location_column_names].drop_duplicates().shape[0]
        ############################################################


        ############################################################
        # create a one-hot-vector version for df_training_categorical
        self.df_training_one_hot_vector = self.apply_one_hot_vector_encoding(df_categorical=self.df_training_categorical)
        ############################################################


