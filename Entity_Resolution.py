import json
import csv
import numpy as np
import pandas as pd
import re

from collections import Counter
from fuzzywuzzy import fuzz
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer

"""
This assignment can be done in groups of 3 students. Everyone must submit individually.

Write down the UNIs of your group (if applicable)

Name : Roland Zhou
Uni  : rz2388

Member 2: Mohammad Radiyat, mr3719

Member 3: Kathy Lin, kl2615
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
    train_df = crossjoin(foursq_train,
        locu_train,
        suffixes=['_foursq', '_locu'])

    # Merge the match labels and training data
    matches_train.columns = ['id_locu', 'id_foursq']
    matches_train['matched'] = 1
    train_df = train_df.merge(matches_train,
        how='left',
        left_on=['id_locu', 'id_foursq'],
        right_on=['id_locu', 'id_foursq'])
    train_df['matched'].fillna(0, inplace=True)

    foursq_test = pd.read_json(foursquare_test_path)
    locu_test = pd.read_json(locu_test_path)

    # Join the foursq and locu data using a cartesian product
    test_df = crossjoin(foursq_test,
        locu_test,
        suffixes=['_foursq', '_locu'])

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

    '''
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

    df_train, _ = filter_matches(train_df,
        lambda df: simple_addr_match(df) & name_match(df))
    df_test, filtered_matches = filter_matches(test_df,
        lambda df: simple_addr_match(df) & name_match(df))

    df_train = filter_neg(df_train,
        lambda df: ~postal_match(df) & ~region_match(df))
    df_test = filter_neg(df_test,
        lambda df: ~postal_match(df) & ~region_match(df))

    feats_train = featurize(df_train,
                            lat_long_continuous,
                            name_fuzzy,
                            addr_fuzzy,
                            phone_num_match)

    feats_test = featurize(df_test,
                           lat_long_continuous,
                           name_fuzzy,
                           addr_fuzzy,
                           phone_num_match)

    '''
    Load and train a gradient boosted tree using LightGBM's
    default configuration and Dataset wrapper API.
    '''
    feats_train_lgb = lgb.Dataset(feats_train, label=df_train['matched'])
    params = {}
    clf = lgb.train(params, feats_train_lgb, 100)

    fake_preds = [True] * _.shape[0]
    print('F1 score: {}'.format(
        f1_score(np.concatenate([clf.predict(feats_train) > .35, fake_preds]),
                 np.concatenate([df_train['matched'].values > 0, fake_preds]))
    ))

    # Binarize continuous prediction scores from 0 to 1
    preds_test = clf.predict(feats_test) > .35
    print("Shapes:")
    print(feats_test.shape)
    print(preds_test.shape)
    print(df_test.shape)

    # Select rows which are matches via boolean indexing and write to csv
    matches_test = df_test[preds_test][['id_locu', 'id_foursq']]
    matches_test = pd.concat([filtered_matches[['id_locu', 'id_foursq']], matches_test])
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

def simple_addr_match(df):
    '''Simple address exact matching for filtering purposes'''
    return ~(df['street_address_locu'] == '') \
        & ~(df['street_address_foursq'] == '') \
        & (df['street_address_locu'] == df['street_address_foursq'])

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

'''
Filtering on pairs to avoid training model on all n^2 pairs
'''
def filter_matches(df, *high_prec_feats):
    '''
    Filter out pairs with high precision features.
    Since each location can only be matched once, for each match
    we can reduce all other negative pairs containing that location.
    '''
    # Accumulate the union of positive predictions according to our feats
    # These can be filtered out by virtue of our feats being high precision
    pos = pd.Series([False] * df.shape[0])
    for feat in high_prec_feats:
        pos |= feat(df)

    # Select the positively predicted pairs
    matches = df[pos]

    # Remove all locu and foursquare ids that have already been matched
    # Do this using np.array operations and boolean indexing (which may not be the most efficient)
    keep = pd.Series([True] * (df.shape[0] - matches.shape[0]))
    for id in matches['id_locu']:
        keep = keep & (df['id_locu'] != id)

    for id in matches['id_foursq']:
        keep = keep & (df['id_foursq'] != id)

    # Return the positive predictions separately
    return df[~pos & keep], matches

def filter_neg(df, *high_rec_feats):
    '''
    Filter out pairs that do not match against very rec features.
    This means we can exclude them from consideration because they are
    unlikely to match.
    '''
    excl = pd.Series([False] * df.shape[0])
    for feat in high_rec_feats:
        excl |= feat(df)

    return df[~excl]

'''Evaluating precision and recall of potentially simple filtering heuristics'''
def postal_match(df):
    return df['postal_code_locu'] == df['postal_code_foursq']

def region_match(df):
    return df['locality_locu'] == df['locality_foursq']

def addr_match(df):
    return ~(df['street_address_locu'] == '') \
        & ~(df['street_address_foursq'] == '') \
        & (df['street_address_locu'] == df['street_address_foursq'])

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
