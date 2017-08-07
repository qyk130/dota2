# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:06:17 2017

@author: Kai 3164421570
"""

import dota2api
import pandas as pd
import numpy as np
#%%
api = dota2api.Initialise(raw_mode=True)
mid = 3164421570
match = api.get_match_details(match_id = mid)
player = pd.DataFrame(match['players'])
player['match_id'] = match['match_id']
players = player
del match['players']
matches = pd.DataFrame(match, index = [0])

#%%
i = 1
while i < 100000:
    mid -=1
    try:
        match = api.get_match_details(match_id = mid)
        if (match['game_mode'] == 22):
            if ('radiant_win' in match):
                player = pd.DataFrame(match['players'])
                player['match_id'] = match['match_id']
                players = players.append(player)
                del match['players']
                match = pd.DataFrame(match, index = [i])
                matches = matches.append(match) 
                i+=1
    except dota2api.src.exceptions.APIError:
        continue
#%%
matches = pd.read_csv('match2.csv')[['match_id','radiant_win']]
players = pd.read_csv('players2.csv')[['account_id','hero_id','gold_per_min','xp_per_min','kills','deaths','assists']]
matches = matches[matches['match_id'] != 3164421570]
players = players[players['match_id'] != 3164421570]
#%%
players.to_csv('players2.csv')
#%%
players = pd.read_csv('players2.csv')
players.loc[players['account_id'] == 4294967295, 'account_id'] = 0
#%%
api = dota2api.Initialise(raw_mode=True)
heroes = api.get_heroes();
heroes = pd.DataFrame(heroes['heroes'])[['id','localized_name']]
heroes.to_csv('hero_names.csv')