# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:58:59 2021

@author: Benjamin
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('../data/all_everest_and_members.csv')

# data checks

status_possible = df['status'].unique()  # about 200 needs to be defined better

occupation_possible = df['occupation'].unique()

nation_possible = df['nation'].unique()

# boolean to binary

boolean_cols = ['leader', 'support', 'disabled', 'hired', 'sherpa', 'tibetan',
                'deputy', 'msolo', 'mtraverse', 'mski', 'mparapente', 'mspeed',
                'mo2none', 'comrte', 'stdrte', 'nohired', 'death', 'msuccess',
                'mo2climb', 'mo2descent', 'mo2sleep', 'mo2medical', 'mclaimed',
                'mdisputed'
                ]

df[boolean_cols] = df[boolean_cols] * 1

# leader/exped info

index = df.index
cond_leaders_only = df['leader'] == 1
leaders_only = index[cond_leaders_only]
leaders_only = df[df.index.isin(leaders_only)]

# climber nation = expedition nation feature

def citizen_is_nation(row):
    citizenships = row.citizen.split('/')
    if row.nation == row.citizen:
        return 1
    elif row.nation in citizenships:
        return 1
    else:
        return 0


df['citizen_is_nation'] = df.apply(citizen_is_nation, axis=1)

# percentage team hired

df = df.drop(df[df.totmembers == 0].index)  # bc only and 2 sherpa fixing rope
df['hired_ratio'] = df.apply(lambda x: x.tothired / x.totmembers, axis=1)

# climber nation = expedition leaders nation feature

def climber_leader_citizen(row):
    citizenships = row.citizen.split('/')
    leader = (leaders_only[leaders_only['expdid'] == row.expdid])
    if leader.empty:
        return row.citizen_is_nation
    # they tend to be all the same nation bar hired so no nationality conflict
    leader_citizenships = str(leader.citizen.iloc[0]).split('/')
    
    for citizen in citizenships:
        for l_citizen in leader_citizenships:
            if citizen == l_citizen:
                return 1
    return 0


df['same_nat_as_leader'] = df.apply(climber_leader_citizen, axis='columns')

# percentage team is same nation feature sans hired

# occupations into categories

# dealing with claimed and disputed

# season is 1 spring, 2 summer ...
# sorting false and true into 1 and 0

# dropping unnessicary information

df = df.drop(['membid', 'peakid', 'age', 'birthdate', 'bcdate',
              'yob', 'residence', 'mhighpt', 'mperhighpt', 'msmtdate1',
              'msmtdate2', 'msmtdate3', 'msmttime1', 'msmttime2', 'msmttime3',
              'mroute1', 'mroute2', 'mroute3', 'mascent1', 'mascent2',
              'mascent3', 'mo2used', 'deathdate', 'deathtime', 'deathtype',
              'deathhgtm', 'deathclass', 'msmtbid', 'msmtterm', 'mchksum',
              'leaders', 'mdeaths', 'pkname', 'heightm', 'smthired',
              'smtmembers', 'hdeaths', 'mo2note'],
             axis=1)

index = df.index
cond_not_to_basecamp = df['nottobc'] == True
not_to_basecamp = index[cond_not_to_basecamp]
df = df.drop(not_to_basecamp, axis=0)
df = df.drop(['nottobc'], axis=1)

index = df.index
cond_basecamp_only = df['bconly'] == True
basecamp_only = index[cond_basecamp_only]
df = df.drop(basecamp_only, axis=0)
df = df.drop(['bconly'], axis=1)

# adding non standard routes in

index = df.index
cond_standard = df['stdrte'] == 0
standard_only = index[cond_standard]

def non_standard_map(row):
    if row.stdrte == 0:
        return 'non standard route'
    else:
        return row.route1

df['route1'] = df.apply(non_standard_map, axis='columns')
df = df.drop(['stdrte', 'comrte'], axis=1)

df = df[df['route1'].notna()]
df['route1'] = df.apply(lambda x: x.route1.lower().split(' (')[0], axis=1)
df['route1'] = df.apply(lambda x: x.route1.replace('/', '-'), axis=1)

route_possible = df['route1'].unique()

# hot oneing categorical data

cat_cols = ['mseason', 'sex', 'citizen', 'status', 'termreason', 'nation',
            'route1']

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(df[cat_cols]))
OH_cols.index = df.index
num_data = df.drop(cat_cols, axis=1)
df_hot_oned = pd.concat([num_data, OH_cols], axis=1)

df = df.drop(['fname', 'lname', 'status', 'occupation', 'route2', 'route3',
              'route4', 'sponsor', 'termreason', 'expdid'],
             axis=1)

df.to_csv('../data/processed_data.csv')

df_hot_oned = df_hot_oned.drop(['fname', 'lname', 'occupation', 'route2',
                                'route3', 'route4', 'sponsor', 'expdid'],
                               axis=1)

df_hot_oned.to_csv('../data/fully_processed_data.csv')

# modern age of climbing data only

df_hot_oned = df_hot_oned.drop(df.index[df.myear < 1990], axis = 0)

df_hot_oned = df_hot_oned.drop(['myear', 'hcn'], axis = 1)

df_hot_oned.to_csv('../data/modern_no_year.csv')
