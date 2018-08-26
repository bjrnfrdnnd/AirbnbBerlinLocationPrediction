# data downloading and cleaning
import os
import shutil
import urllib.request
import zipfile
import glob
import re
import pandas as pd
import numpy as np

def delete_path(filepath: str):
    """delete the path pointed to by filepath


        :param filepath: the path that is to be deleted
        :type filepath: str
        :return: None
        :rtype: None
    """
    if os.path.islink(filepath):
        os.unlink(filepath)
    elif os.path.isdir(filepath):
        shutil.rmtree(filepath)
    elif os.path.isfile(filepath):
        os.remove(filepath)


def create_dir(dir_name: str,
               overwrite:bool=False):
    """create the directory pointed to by dir_name

        If dir_name does not exist, create it
        If overwrite == True, delete dir_name if it exists and then recreate the dir
        If ovverwrite == False, do not delete dir_name; this means that if
           dir_name exists, it could be a file and not a dir

        :param dir_name: the path to the directory that is to be created
        :param overwrite: whether or not to delete dir_name prior to creation of the dir

        :type dir_name: str
        :type overwrite: bool

        :return: None
        :rtype: None
    """
    # overwrite will recreate the data directory
    if overwrite:
        delete_path(dir_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

class source_dataClass(object):
    def __init__(self,
                 work_dir:str=None,
                 source_data_dir:str='source_data_dir',
                 combined_csv_filename:str='combined.csv',
                 source_url:str='https://s3.amazonaws.com/tomslee-airbnb-data-2/berlin.zip',
                 date_column_name:str='last_modified',
                 ):

        if work_dir is None:
            work_dir = os.path.join('.', 'work_dir_for_AirbnbBerlinLocationPrediction')
            work_dir = os.path.realpath(work_dir)

        if not os.path.isabs(work_dir):
            work_dir = os.path.realpath(work_dir)
        self._work_dir = work_dir

        if not os.path.isabs(source_data_dir):
            source_data_dir = os.path.realpath(os.path.join(self._work_dir,source_data_dir))
        self._source_data_dir = source_data_dir

        if not os.path.isabs(combined_csv_filename):
            combined_csv_filename = os.path.realpath(os.path.join(self._source_data_dir,combined_csv_filename))
        self._combined_csv_filename = combined_csv_filename


        # default configurations
        self._overwrite_data_dir = False
        self._overwrite_zip_file = False
        self._unzip_zip_file = False

        # filenames and dirnames for downloaded files
        self._target_zip_filename = os.path.join(self._source_data_dir,'berlin.zip')
        self._source_url = source_url
        self._unzipped_csv_files_dir = os.path.join(self._source_data_dir,'s3_files','berlin')
        self._date_column_name = date_column_name
        self._survey_id_column_name = 'survey_id'

        # csv filename that are created
        self._combined_csv_filename = os.path.realpath(os.path.join(self._source_data_dir,'berlin_combined.csv'))
        self._clean_csv_filename = os.path.realpath(os.path.join(self._source_data_dir,'berlin_clean.csv'))

        # we will keep the dataframe corresponding to the clean csv in memory
        self._df_clean_csv = None


        # the dataframe

        # print(self._work_dir)
        # print(self._source_data_dir)
        # print(self._target_zip_filename)
        # print(self._source_url)
        # print(self._unzipped_csv_files_dir)

    def download_and_clean(self):
        """convenience function to download, unzip, combine and clean the source data

        :return: clean source data dataframe
        :rtype: pd.DataFrame
        """

        # download and unzip the data
        self.create_source_datadir()
        self.download_zip_file()
        self.unzip_data()


        # combine csv files to one csv file and fill missing survey_ID
        # this also writes the resulting csv file to disk
        self.combine_source_csv_files()

        # clean the combined csv file
        df_clean_csv = self.clean_combined_csv_file()

        # save cleaned csv
        df_clean_csv.to_csv(self._clean_csv_filename, index=False)

        # read the cleaned csv into memory
        df_clean_csv = self.read_clean_csv()

        return df_clean_csv

    def download_zip_file(self,
                          overwrite:bool=None):
        """download Airbnb data for Berlin

            :param overwrite: whether or not to delete the target zip file (self._target_zip_filename) and redownload it

            :type overwrite: bool

            :return: None
            :rtype: None
        """
        target_filename = self._target_zip_filename
        data_url = self._source_url

        overwrite = self._overwrite_zip_file if overwrite is None else overwrite

        if overwrite:
            delete_path(filepath=target_filename)

        if not os.path.exists(target_filename):
            urllib.request.urlretrieve(data_url, target_filename)

    def create_source_datadir(self,
                              overwrite:bool=None):
        """create the source data directory to which we download and unzip the original data

            :param overwrite: whether or not to delete the target dir and recreate it

            :type overwrite: bool

            :return: None
            :rtype: None
        """

        overwrite = self._overwrite_data_dir if overwrite is None else overwrite

        create_dir(
            dir_name=self._source_data_dir,
            overwrite=overwrite)

    def unzip_data(self,
                   overwrite:bool=None):
        """unzip the downloaded zip file

        This always overwrites the old files

        :return:
        :rtype:
        """

        overwrite = self._overwrite_zip_file if overwrite is None else overwrite

        data_dir = self._source_data_dir
        source_data_filename = self._target_zip_filename

        if overwrite or len(glob.glob(os.path.join(self._unzipped_csv_files_dir,'*csv')))==0:
            zip_ref = zipfile.ZipFile(source_data_filename, 'r')

            zip_ref.extractall(data_dir)
            zip_ref.close()



    def _read_one_source_csv_file(self, filename:str):
        """Reads one csv file located at filename. Corrects missing survey_id


        :param filename: the filename of the csv file to read

        :type filename: str

        :return:
        :rtype:
        """

        date_column_names = [self._date_column_name]
        survey_id_column_name = self._survey_id_column_name

        m = re.search('.*(\d{4})_\d{4}.*', filename)
        survey_id = m.group(1) if m else np.NaN
        survey_id = f'{survey_id}'

        df = pd.read_csv(filename, parse_dates=date_column_names)
        if survey_id_column_name not in df.columns:
            df[survey_id_column_name] = survey_id
        if 'name' not in df.columns:
            df['name'] = ""

        return df

    def combine_source_csv_files(self):
        """read all source csv files and combine them to a single one

        :return:
        :rtype:
        """
        source_csv_directory = self._unzipped_csv_files_dir
        csv_file_list = glob.glob(os.path.join(source_csv_directory, '*.csv'))

        # the final dataframe
        df = None

        for filename in csv_file_list:
            df_ = self._read_one_source_csv_file(filename=filename)
            if df is None:
                df = df_
            else:
                try:
                    df = pd.concat([df, df_], sort=False)
                except:
                    df = pd.concat([df, df_])

        df.sort_values(by=[self._date_column_name],ascending=False,inplace=True)

        df.to_csv(self._combined_csv_filename,index=False)

    def clean_combined_csv_file(self):
        """ clean the combined csv file

        * drop non-needed columns
        * fill missing values
        * add year and yearmonth information

        :return:
        :rtype:
        """
        date_column_name = self._date_column_name

        # add year and yearmonth
        df = pd.read_csv(self._combined_csv_filename,
                         parse_dates=[self._date_column_name],
                         dtype={
                             'name':str,
                             self._survey_id_column_name:str,
                             'city':str,
                             'location':str
                         })

        df['year'] = df[date_column_name].dt.year
        df['yearmonth'] = df[date_column_name].map(lambda x: 100 * x.year + x.month)

        yearmonthIndex = np.sort(df['yearmonth'].unique())
        df['yearmonthIndex'] = df['yearmonth'].map(lambda x: np.nonzero(yearmonthIndex == x)[0][0])

        # drop all not needed columns
        colsToKeep = [
            'price',
            'room_type',
            'accommodates',
            'bedrooms',
            'latitude',
            'longitude',
            'last_modified',
            'year',
            'yearmonth',
            'yearmonthIndex',
            'survey_id']
        df.drop(columns=df.columns.difference(colsToKeep), inplace=True)
        df.drop_duplicates(inplace=True)

        # reset index
        df.reset_index(drop=True, inplace=True)

        # fill missing information in the bedroom column with the most common value
        df['bedrooms'].fillna(df['bedrooms'].mode()[0], inplace=True)

        # fill missing information in the accommodates column with the most common value
        df['accommodates'].fillna(df['accommodates'].mode()[0], inplace=True)

        # consider missing information in the room type column as 'Not given'
        df['room_type'] = df['room_type'].fillna('Not given')

        return df


    def read_clean_csv(self):
        """read the clean csv into a pandas dataframe for additional manipulation

        :return:
        :rtype:
        """
        df = pd.read_csv(self._clean_csv_filename,parse_dates=[self._date_column_name])
        return df

