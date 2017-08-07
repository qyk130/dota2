# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import ensemble 
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# this is meant to be a simple example so only matches and players are used
matches = pd.read_csv('match.csv', index_col=0)
players = pd.read_csv('players.csv')[['account_id','hero_id','gold_per_min','xp_per_min','kills','deaths','assists']]
test_player = pd.read_csv('test_player.csv').head(100000)
test_labels = pd.read_csv('test_labels.csv')['radiant_win'].head(10000).astype(int)
#%%
train_labels = matches['radiant_win'].astype(int)
ids_train = players[['account_id','hero_id']].values.reshape(50000,10,2)
ids_test = test_player[['account_id','hero_id']].values.reshape(10000,10,2)
player_score = players.groupby(['account_id','hero_id']).mean().reset_index()
player_score['deaths'] = player_score['deaths'].replace([0],1)
player_score['kda'] = (player_score['kills']+player_score['assists'])/player_score['deaths']
player_score.drop(['deaths','assists','kills'], axis=1, inplace=True)
player_score['kda'].fillna(0, inplace=True)
player_score = player_score.drop_duplicates()
player_average = player_score.copy()
player_average.drop('account_id', 1,inplace=True)
player_average = player_average.groupby('hero_id').mean
x_train = np.zeros((50000,678))
x_test = np.zeros((10000,678))
#%%
failed_count = 0
for i in range(0,50000):
    for j in range(0,10):
        if j < 5:
            offset = 0
        if j >=5:
            offset = 339
        account = ids_train[i,j,0]
        hero = ids_train[i,j,1]
        info = player_score[(player_score['account_id'] == 0) & (player_score['hero_id'] == hero)][['gold_per_min','xp_per_min','kda']]
        x_train[i,(hero-1)*3 + offset] = info.values[0,0]
        x_train[i,(hero-1)*3 + offset + 1] = info.values[0,1]
        x_train[i,(hero-1)*3 + offset + 2] = info.values[0,2]
for i in range(0,10000):
    for j in range(0,10):
        if j < 5:
            offset = 0
        if j >=5:
            offset = 339
        account = ids_test[i,j,0]
        hero = ids_test[i,j,1]
        if account>158360:
            account = 0
            failed_count+=1
        info = player_score[(player_score['account_id'] == 0) & (player_score['hero_id'] == hero)][['gold_per_min','xp_per_min','kda']]
        if len(info) == 0:
            failed_count+=1
            info = player_score[(player_score['account_id'] == 0) & (player_score['hero_id'] == hero)][['gold_per_min','xp_per_min','kda']]
        x_test[i,(hero-1)*3 + offset] = info.values[0,0]
        x_test[i,(hero-1)*3 + offset + 1] = info.values[0,1]
        x_test[i,(hero-1)*3 + offset + 2] = info.values[0,2]

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
y = matches['radiant_win'].values
logreg = LogisticRegression()
logreg.fit(x_train, train_labels)
acc_scorer = make_scorer(accuracy_score)
predictions = logreg.predict(x_test)
print(accuracy_score(test_labels, predictions))