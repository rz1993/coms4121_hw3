import json
import csv
import numpy as np
import pandas as pd
import re

from collections import Counter
from fuzzywuzzy import fuzz
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

"""
This assignment can be done in groups of 3 students. Everyone must submit individually.

Write down the UNIs of your group (if applicable)

Name : your name
Uni  : your uni

Member 2: name, uni

Member 3: name, uni
"""
def get_matches(locu_train_path, foursquare_train_path, matches_train_path, locu_test_path, foursquare_test_path):
    """
        In this function, You need to design your own algorithm or model to find the matches and generate
        a matches_test.csv in the current folder.

        you are given locu_train, foursquare_train json file path and matches_train.csv path to train
        your model or algorithm.

        Then you should test your model or algorithm with locu_test and foursquare_test json file.
        Make sure that you write the test matches to a file in the same directory called matches_test.csv.

    """

    # Load the training data as pandas dataframes
    matches_train = pd.read_csv(matches_train_path)
    foursq_train = pd.read_json(foursquare_train_path)
    locu_train = pd.read_json(locu_train_path)

    # Join the foursq and locu data using a cartesian product
    train_df = crossjoin(foursq_train, locu_train, suffixes=['_foursq', '_locu'])

    # Merge the match labels and training data
    matches_train.columns = ['id_locu', 'id_foursq']
    matches_train['matched'] = 1
    train_df = train_df.merge(matches_train, how='left', left_on=['id_locu', 'id_foursq'], right_on=['id_locu', 'id_foursq'])
    train_df['matched'].fillna(0, inplace=True)

    foursq_test = pd.read_json(foursquare_test_path)
    locu_test = pd.read_json(locu_train_path)

    # Join the foursq and locu data using a cartesian product
    test_df = crossjoin(foursq_test, locu_test, suffixes=['_foursq', '_locu'])

    # Gathering common brand names based on counts
    all_names = foursq_train['name'] \
                .append(locu_train['name']) \
                .append(foursq_test['name']) \
                .append(locu_test['name'])
    all_names.apply(lambda n: norm_name(n, default='None'))
    name_counts = Counter(all_names)
    brand_names = [name for name in name_counts if name_counts[name] > 2]

    def name_match(df):
        return name_match_no_brands(df, brand_names)

    features_train = featurize(train_df,
                               lat_long_continuous,
                               address_match,
                               name_match,
                               name_fuzzy,
                               addr_fuzzy,
                               phone_num_match)
    features_test = featurize(test_df,
                              lat_long_continuous,
                              address_match,
                              name_match,
                              name_fuzzy,
                              addr_fuzzy,
                              phone_num_match)

    '''
    Load and train a gradient boosted tree using LightGBM's
    default configuration and Dataset wrapper API.
    '''
    features_train_lgb = lgb.Dataset(features_train, label=train_df['matched'])
    params = {}
    clf = lgb.train(params, features_train_lgb, 100)

    # Binarize continuous prediction scores from 0 to 1
    preds_test = clf.predict(features_test)
    preds_test = preds_test > 0.35

    # Select rows which are matches via boolean indexing and write to csv
    matches_test = test_df.loc[preds_test][['id_locu', 'id_foursq']]
    matches_test.columns = ['locu_id', 'foursquare_id']
    matches_test.to_csv('matches_test.csv', index=False)


def crossjoin(df1, df2, **kwargs):
    # Cross join the two org's data to form a dataframe of all possible pairs
    df1['_tmp'] = 1
    df2['_tmp'] = 1
    joined = pd.merge(df1, df2, on=['_tmp'], **kwargs).drop('_tmp', axis=1)
    return joined

'''
Text processing
'''

stop_words = ['restaurant', 'inc', 'cafe', 'bakery', 'and', 'the', 'of']

def remove_words(s, words):
    return ' '.join([token for token in s.split() if s not in words])

def remove_punc(s):
    return re.sub('[.,&#\'()]', '', s)

def norm_name(name, default=''):
    # Normalize a location's name to get matches
    if not name:
        return default

    name = str(name).lower()
    name = remove_punc(name)
    name = remove_words(name, stop_words)
    return name

def unigrams(name):
    return norm_name(name).split()

'''
Discrete Features
'''

def lat_long_match(df):
    # Latitude and longitudes differ in decimal places for foursquare and locu
    # Errors in coordinates and misisng values also reduce recall
    def norm_coord(coord):
        return round(coord, 4)
    return (df['latitude_foursq'].apply(norm_coord) == df['latitude_locu'].apply(norm_coord)) \
         & (df['longitude_foursq'].apply(norm_coord) == df['longitude_locu'].apply(norm_coord))

def phone_num_match(df):
    def norm_foursq_phone(p, default='None'):
        if not p:
            return default
        p = str(p).replace('(', '')
        p = p.replace(')', '')
        p = p.replace(' ', '')
        p = p.replace('-', '')
        return p

    norm_phone = df['phone_foursq'].apply(lambda p: norm_foursq_phone(p, default='None_foursq'))
    return norm_phone == df['phone_locu'].apply(lambda p: norm_foursq_phone(p, default='None_locu'))

def address_match(df):
    def norm(a, default='None'):
        if not a:
            return default
        a = str(a).lower()
        a = a.replace('east', 'e')
        a = a.replace('west', 'w')
        a = a.replace('south', 's')
        a = a.replace('north', 'n')
        a = a.replace('square', 'sq')
        a = a.replace('th ', ' ')
        a = a.replace('st ', ' ')
        a = a.replace('st. ', ' ')
        a = a.replace('.', '')
        a = a.replace(',', '')
        return a

    return df['street_address_foursq'].apply(lambda a: norm(a, default='None_foursq')) \
        == df['street_address_locu'].apply(lambda a: norm(a, default='None_locu'))

def name_match_no_brands(df, brand_names):
    normed1 = df['name_foursq'].apply(lambda a: norm_name(a, default='None_foursq'))
    normed2 = df['name_locu'].apply(lambda a: norm_name(a, default='None_locu'))
    return ((normed1 == normed2)
        & pd.Series([name not in brand_names for name in df['name_foursq']])
        & pd.Series([name not in brand_names for name in df['name_locu']]))

'''
Continuous Features (mostly based on fuzzy matching)
'''
def name_fuzzy(df):
    fuzzy_ratio = df.apply(lambda row: fuzz.ratio(norm_name(row['name_foursq'], default='None_foursq'),
                                                  norm_name(row['name_locu'], default='None_locu')), axis=1)
    return fuzzy_ratio

def addr_fuzzy(df):
    fuzzy_ratio = df.apply(lambda row: fuzz.ratio(row['street_address_foursq'] or 'None_foursq',
                                                  row['street_address_locu'] or 'None_locu'), axis=1)
    return fuzzy_ratio

def lat_long_continuous(df):
    # Latitude and longitudes differ in decimal places for foursquare and locu
    # Errors in coordinates and misisng values also reduce recall
    def norm_coord(coord):
        return round(coord, 4) if not np.isnan(coord) else -1

    def euclid_dist(row):
        return np.sqrt((norm_coord(row['latitude_foursq'])
                        - norm_coord(row['latitude_locu']))**2
                       + (norm_coord(row['longitude_foursq'])
                        - norm_coord(row['longitude_locu']))**2)

    return df.apply(euclid_dist, axis=1)

'''Creating the feature matrix'''

def featurize(df, *features):
    feat_df = pd.DataFrame({f.__name__: f(df) for f in features})
    return feat_df

if __name__ == '__main__':
    get_matches('train/locu_train.json',
                'train/foursquare_train.json',
                'train/matches_train.csv',
                'online_competition/locu_test.json',
                'online_competition/foursquare_test.json')
