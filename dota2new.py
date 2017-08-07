# -*- coding: utf-8 -*-
"""
Created on Sun May  7 19:27:12 2017

@author: Kai
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import ensemble 
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# this is meant to be a simple example so only matches and players are used
train_size = 50000
test_size = 100000
players = pd.read_csv('players.csv')[['account_id','hero_id','gold_per_min','xp_per_min','kills','deaths','assists']]
test_player = pd.read_csv('test_player.csv')[['account_id','hero_id']]
players = players.head(train_size * 10)
train_labels = pd.read_csv('match.csv')['radiant_win']
test_labels = pd.read_csv('test_labels.csv')['radiant_win']
train_labels = train_labels.head(train_size).astype(int).values
hero_names = pd.read_csv('hero_names.csv')[['hero_id','localized_name']]
players['hero_id'] = players['hero_id'].apply(lambda id: (id - 1) if id <= 24 else (id - 2))
test_player['hero_id'] = test_player['hero_id'].apply(lambda id: (id - 1) if id <= 24 else (id - 2))
#%%
hero_count = 113
total_count = (hero_count) * 2
synergy = np.zeros((hero_count,hero_count,3))
counter = np.zeros((hero_count,hero_count,3))
hero_train = players['hero_id'].values.reshape(train_size,10)
hero_test = test_player['hero_id'].values.reshape(test_size,10)
for i in range(0,train_size):
    for j in range(0,10):
        if j < 5:
            for k in range(j + 1, 5):
                hero1 = min(hero_train[i,j],hero_train[i,k]);
                hero2 = max(hero_train[i,j],hero_train[i,k]);
                synergy[hero1,hero2,0] += train_labels[i]
                synergy[hero1,hero2,1] += 1
            for k in range(5, 10):
                counter[hero_train[i,j],hero_train[i,k],0] += train_labels[i]
                counter[hero_train[i,j],hero_train[i,k],1] += 1
                counter[hero_train[i,k],hero_train[i,j],0] += 1 - train_labels[i]
                counter[hero_train[i,k],hero_train[i,j],1] += 1
        if j >= 5:
            for k in range(j + 1, 10):
                hero1 = min(hero_train[i,j],hero_train[i,k]);
                hero2 = max(hero_train[i,j],hero_train[i,k]);
                synergy[hero1,hero2,0] += 1 - train_labels[i]
                synergy[hero1,hero2,1] += 1
for i in range(0, hero_count):
    for j in range(0, hero_count):
        if (counter[i,j,0] == 0) | (counter[i,j,1] == 0):
            counter[i,j,2] = 0.5
        else:
            counter[i,j,2] = counter[i,j,0]/counter[i,j,1]
        if (synergy[i,j,0] == 0) | (synergy[i,j,1] == 0):
            synergy[i,j,2] = 0.5
        else:
            synergy[i,j,2] = synergy[i,j,0]/synergy[i,j,1]
        
#%%
threshold_low = 0.6
threshold_high = 0.8
min_sup = 100
x_train = np.zeros((train_size,total_count))
x_test = np.zeros((test_size,total_count))
train_synergy_feat = np.zeros((train_size,2*hero_count*hero_count),dtype=int)
train_counter_feat = np.zeros((train_size,2*hero_count*hero_count),dtype=int)
test_synergy_feat = np.zeros((test_size,2*hero_count*hero_count),dtype=int)
test_counter_feat = np.zeros((test_size,2*hero_count*hero_count),dtype=int)
def check_synergy(data,hero1,hero2,radiant):
    if (hero1 > hero2):
        hero1, hero2 = hero2, hero1
    if (threshold_low<=synergy[hero1,hero2,2]) & (synergy[hero1,hero2,2]<=threshold_high) & (synergy[hero1,hero2,1] >= min_sup):
        if radiant == 0:
            data[i,hero1*hero_count + hero2] = 1
        else:
            data[i,hero_count * hero_count + hero1*hero_count + hero2] = 1
def check_counter(data,hero1,hero2,radiant):
    if (threshold_low<=counter[hero1,hero2,2]) & (counter[hero1,hero2,2]<=threshold_high) & (counter[hero1,hero2,1] >= min_sup):
        if radiant == 0:
            data[i,hero1*hero_count + hero2] = 1
        else:
            data[i,hero_count * hero_count + hero1*hero_count + hero2] = 1
for i in range(0,train_size):
    for j in range(0,10):
        if j < 5:
            x_train[i,hero_train[i,j]] = 1
            for k in range(j+1,5): 
                check_synergy(train_synergy_feat, hero_train[i,j], hero_train[i,k], 0)
            for k in range(5,10):
                check_counter(train_counter_feat, hero_train[i,j], hero_train[i,k], 0)
        if j >=5:
            x_train[i,hero_train[i,j] + hero_count] = 1
            for k in range(j+1,10): 
                check_synergy(train_synergy_feat, hero_train[i,j], hero_train[i,k], 1)
            for k in range(0,5):
                check_counter(train_counter_feat, hero_train[i,j], hero_train[i,k], 1)
for i in range(0,test_size):
    for j in range(0,10):
        if j < 5:
            x_test[i,hero_test[i,j]] = 1
            for k in range(0,5): 
                check_synergy(test_synergy_feat, hero_test[i,j], hero_test[i,k], 0)                   
            for k in range(5,10):
                check_counter(test_synergy_feat, hero_test[i,j], hero_test[i,k], 0)   
        if j >=5:
            x_test[i,hero_test[i,j] + hero_count] = 1
            for k in range(5,10): 
                check_synergy(test_synergy_feat, hero_test[i,j], hero_test[i,k], 0)  
            for k in range(0,5):
                check_counter(test_synergy_feat, hero_test[i,j], hero_test[i,k], 0) 
#%%
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import StandardScaler 
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import optimizers, metrics, backend as K
batch_size = 200
epsilon_std = 0.1
latent_dim = 8
x = Input(shape=(113,))
h = Dense(256, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_std = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_std = args
    epsilon = K.random_normal(shape=(batch_size,latent_dim), mean=0., stddev = epsilon_std)
    return z_mean + K.exp(z_std / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean,z_std])

decoder_h = Dense(256, activation='relu')
decoder_mean = Dense(113, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

vae = Model(x, x_decoded_mean)
encoder = Model(x, z_mean)

def vae_loss(x, x_decoded_mean):
    xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_std - K.square(z_mean) - K.exp(z_std), axis=-1)
    return xent_loss + kl_loss
    
vae.compile(optimizer='adam', loss=vae_loss)
train_vert = np.vstack((x_train[:,:113],x_train[:,113:226]))
test_vert = np.vstack((x_test[:,:113],x_test[:,113:226]))
vae.fit(train_vert,train_vert,epochs=500,
                batch_size=batch_size,
                shuffle=True)
train_encoded = np.vsplit(encoder.predict(train_vert),2)
train_encoded = np.concatenate((train_encoded[0],train_encoded[1]), axis=1)
test_encoded = np.vsplit(encoder.predict(test_vert),2)
test_encoded = np.concatenate((test_encoded[0],test_encoded[1]), axis=1)
#%%
logreg = LogisticRegression(max_iter = 200,)
logreg.fit(x_train, train_labels)
acc_scorer = make_scorer(accuracy_score)
predictions = logreg.predict(x_train)
print(accuracy_score(train_labels,predictions))
predictions = logreg.predict(x_test)
print(accuracy_score(test_labels,predictions))
#%%
clf = MLPClassifier(solver='adam', alpha=1e-7,hidden_layer_sizes=(50,20,2), random_state=870,activation='relu',momentum = 0.1,learning_rate_init = 0.05, batch_size = 200,max_iter = 500, early_stopping = False, verbose = True)
clf.fit(train_encoded, train_labels)
acc_scorer = make_scorer(accuracy_score)
predictions = clf.predict(train_encoded)
print(accuracy_score(train_labels,predictions))
predictions = clf.predict(test_encoded)
print(accuracy_score(test_labels,predictions))
#%%
from sklearn.svm import SVC
svm = SVC(C=0.25,kernel='rbf')
svm.fit(train_encoded, train_labels)
acc_scorer = make_scorer(accuracy_score)
predictions = svm.predict(train_encoded)
print(accuracy_score(train_labels,predictions))
predictions = svm.predict(test_encoded)
print(accuracy_score(test_labels,predictions))
#%%
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(base_estimator=svm,n_estimators=5,learning_rate=0.5,algorithm="SAMME")
ada.fit(x_train[:,:230], train_labels)
acc_scorer = make_scorer(accuracy_score)
predictions = ada.predict(x_train[:,:230])
print(accuracy_score(train_labels,predictions))
predictions = ada.predict(x_test[:,:230])
print(accuracy_score(test_labels,predictions))
#%%
base_feat_size = 40
final_feat_size = 60
component_amount = 20
components= [LogisticRegression() for i in range(0,component_amount)]
for i in range(0,component_amount):
    feat_index = np.random.choice(hero_count, base_feat_size, replace=False) 
    feat = np.zeros((train_size,hero_count * 2))
    feat[:,feat_index] = x_train[:,feat_index]
    feat[:,feat_index+hero_count] = x_train[:,feat_index+hero_count]
    components[i] = LogisticRegression(max_iter=200)
    components[i].fit(feat, train_labels)
    predictions = components[i].predict(feat)
    acc = accuracy_score(train_labels,predictions)    
    for j in range(base_feat_size, final_feat_size):    
        feat_iter = (np.delete(range(0,hero_count),feat_index))
        np.random.shuffle(feat_iter)
        for k in feat_iter:
            new_feat = np.empty_like(feat)
            new_feat[:] = feat
            new_feat[:,k] = x_train[:,k]
            new_feat[:,k+hero_count] = x_train[:,k+hero_count]
            components[i] = LogisticRegression(max_iter=200)
            components[i].fit(new_feat, train_labels)
            predictions = components[i].predict(x_train[:,:230])
            new_acc = accuracy_score(train_labels,predictions)
            if (new_acc > acc):
                print(j)
                print(k)
                acc = new_acc
                best_comp = components[i]
                best_feat = new_feat
                best_index = k
                print(acc)
        feat = best_feat
        feat_index = np.append(feat_index,best_index)
        print(feat_index)
    components[i] = best_comp
#%%
combined_result = components[0].predict(x_train[:,:230])
for i in range(1,component_amount):
    combined_result = np.vstack((combined_result,(components[i].predict(x_train[:,:230]))))
combined_result = np.transpose(combined_result)
clf = MLPClassifier(solver='adam', alpha=1e-10,hidden_layer_sizes=(200,2), random_state=54415,activation='relu',momentum = 0.05,learning_rate_init = 0.07, batch_size = 600,max_iter = 500, early_stopping = False, verbose = True)
clf.fit(combined_result, train_labels)
acc_scorer = make_scorer(accuracy_score)
predictions = clf.predict(combined_result)
print(accuracy_score(train_labels,predictions))
combined_result = components[0].predict(x_test[:,:230])
for i in range(1,component_amount):
    combined_result = np.vstack((combined_result,(components[i].predict(x_test[:,:230]))))
combined_result = np.transpose(combined_result)
predictions = clf.predict(combined_result)
print(accuracy_score(test_labels,predictions))
#%%
predictions = components[0].predict(x_train[:,:230])
for i in range(1,component_amount):
    predictions = predictions + components[i].predict(x_train[:,:230])
predictions = predictions/component_amount
predictions = np.around(predictions)
print(accuracy_score(train_labels,predictions))
predictions = components[0].predict(x_test[:,:230])
for i in range(1,component_amount):
    predictions = predictions + components[i].predict(x_test[:,:230])
predictions = predictions/component_amount
predictions = np.around(predictions)
print(accuracy_score(test_labels,predictions))
        
        
    