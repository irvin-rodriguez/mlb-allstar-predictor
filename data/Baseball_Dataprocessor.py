# Given the Lahman Dataset Excel file, outputs clean dataset used for neural network

import pandas as pd
import numpy as np
import pybaseball

# this may take over 2 minutes...

# Define years of interest
years = list(range(2008, 2020)) + list(range(2021, 2024))

# Initialize an empty DataFrame to store player statistics
player_stats_df = pd.DataFrame()

# Fetch player statistics for each year
for year in years:
    stats = pybaseball.batting_stats_range(start_dt=f'{year}-03-01', end_dt=f'{year}-06-30')
    stats['year'] = year   
    player_stats_df = pd.concat([player_stats_df, stats])


# Define a dictionary mapping to normalize names
utf8_to_normalized = {
    r'\xc3\x80': 'A', r'\xc3\x81': 'A', r'\xc3\x82': 'A', r'\xc3\x83': 'A', r'\xc3\x84': 'A',
    r'\xc3\x85': 'A', r'\xc3\x86': 'AE', r'\xc3\x87': 'C', r'\xc3\x88': 'E', r'\xc3\x89': 'E',
    r'\xc3\x8a': 'E', r'\xc3\x8b': 'E', r'\xc3\x8c': 'I', r'\xc3\x8d': 'I', r'\xc3\x8e': 'I',
    r'\xc3\x8f': 'I', r'\xc3\x90': 'D', r'\xc3\x91': 'N', r'\xc3\x92': 'O', r'\xc3\x93': 'O',
    r'\xc3\x94': 'O', r'\xc3\x95': 'O', r'\xc3\x96': 'O', r'\xc3\x98': 'O', r'\xc3\x99': 'U',
    r'\xc3\x9a': 'U', r'\xc3\x9b': 'U', r'\xc3\x9c': 'U', r'\xc3\x9d': 'Y', r'\xc3\x9e': 'TH',
    r'\xc3\x9f': 'ss', r'\xc3\xa0': 'a', r'\xc3\xa1': 'a', r'\xc3\xa2': 'a', r'\xc3\xa3': 'a',
    r'\xc3\xa4': 'a', r'\xc3\xa5': 'a', r'\xc3\xa6': 'ae', r'\xc3\xa7': 'c', r'\xc3\xa8': 'e',
    r'\xc3\xa9': 'e', r'\xc3\xaa': 'e', r'\xc3\xab': 'e', r'\xc3\xac': 'i', r'\xc3\xad': 'i',
    r'\xc3\xae': 'i', r'\xc3\xaf': 'i', r'\xc3\xb0': 'd', r'\xc3\xb1': 'n', r'\xc3\xb2': 'o',
    r'\xc3\xb3': 'o', r'\xc3\xb4': 'o', r'\xc3\xb5': 'o', r'\xc3\xb6': 'o', r'\xc3\xb8': 'o',
    r'\xc3\xb9': 'u', r'\xc3\xba': 'u', r'\xc3\xbb': 'u', r'\xc3\xbc': 'u', r'\xc3\xbd': 'y',
    r'\xc3\xbe': 'th', r'\xc3\xbf': 'y', r"\'": r"'"
}

def replace_with_dictionary(string, dictionary):
    for key, value in dictionary.items():
        string = string.replace(key, value)
    return string

# Iterate through the array and replace UTF-8 literals with normalized forms
array = player_stats_df['Name'].tolist()
for i in range(len(array)):
    array[i] = replace_with_dictionary(array[i], utf8_to_normalized)

player_stats_df['Name'] = array

file = pd.ExcelFile('lahmanData.xlsx')  # Lahman Dataset (provided)

# match seasons to put binary output parameter
all_star = pd.read_excel(file, 'AllstarFull')
people = pd.read_excel(file, 'People')
appearances = pd.read_excel(file, 'Appearances')

intermediate = pd.merge(all_star, people[['playerID', 'nameFirst', 'nameLast', 'birthYear']], on='playerID', how='left')
intermediate['birthYear'] = intermediate['birthYear'].astype(int)
intermediate['A_Name'] = intermediate['nameFirst'] + ' ' + intermediate['nameLast']
intermediate.drop(['nameFirst', 'nameLast'], axis=1, inplace=True)

intermediate.drop(intermediate[intermediate['yearID'] < 1970].index, inplace=True)


games = appearances['G_all']
pitched = appearances['G_p']

# removes pitchers as all-stars (we are doing hitting stats only and they are bad at hitting)
is_pitcher = (pitched / games).round(decimals=0)

appearances['is_pitcher'] = is_pitcher
all_star_df = pd.merge(intermediate, appearances[['playerID', 'yearID', 'is_pitcher']], on=['playerID', 'yearID'], how='left')

all_star_df.drop(all_star_df[all_star_df['is_pitcher'] == 1].index, inplace=True)

s_name = player_stats_df['Name'].tolist()
a_name = all_star_df['A_Name'].tolist()

s_year = player_stats_df['year'].values
a_year = all_star_df['yearID'].values

s_age = player_stats_df['Age'].values
a_birthyear = all_star_df['birthYear'].values

ast = np.zeros(len(s_age))

for i in range(len(s_name)):
    for j in range(len(a_name)):
        if a_name[j] in s_name[i]:
            if s_year[i] == a_year[j]:
                if (s_year[i] - s_age[i]) >= (a_birthyear[j] - 2) and (s_year[i] - s_age[i]) <= (a_birthyear[j] + 2):
                    ast[i] = 1

player_stats_df['All Star'] = ast.astype(int)
# output full dataset
player_stats_df.to_csv('2008_2024_FULL_DATASET.csv', index=False)