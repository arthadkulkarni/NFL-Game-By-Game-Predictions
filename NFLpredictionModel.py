import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#get data
data_2022_updated = pd.read_csv("./Data/BEST_2022_nfl_data_play_with_scores_and_HomeWon (1).csv")
data_2023_updated = pd.read_csv("./Data/BEST_2023_nfl_data_play_with_scores_and_HomeWon (1).csv")
upcomingGames = pd.read_csv("./Data/BEST New_Upcoming_Schedule (1).csv")

#merge data
all_data = pd.concat([data_2022_updated, data_2023_updated])

#classify statistics
avg_points_scored_home = all_data.groupby('Home')['HomeScore'].mean()
avg_points_scored_visitor = all_data.groupby('Visitor')['VisitorScore'].mean()
avg_points_allowed_home = all_data.groupby('Home')['VisitorScore'].mean()
avg_points_allowed_visitor = all_data.groupby('Visitor')['HomeScore'].mean()
overall_avg_points_scored = (avg_points_scored_home + avg_points_scored_visitor) / 2
overall_avg_points_allowed = (avg_points_allowed_home + avg_points_allowed_visitor) / 2
home_wins = all_data.groupby('Home')['HomeWon'].sum()
visitor_wins = all_data.groupby('Visitor').apply(lambda x: len(x) - x['HomeWon'].sum())
total_games_home = all_data['Home'].value_counts()
total_games_visitor = all_data['Visitor'].value_counts()
overall_wins = home_wins + visitor_wins
total_games = total_games_home + total_games_visitor
win_rate = overall_wins / total_games

#define data frame for team statistics
team_features = pd.DataFrame({
    'AvgPointsScored': overall_avg_points_scored,
    'AvgPointsAllowed': overall_avg_points_allowed,
    'WinRate': win_rate
})
team_features.reset_index(inplace=True)
team_features.rename(columns={'Home': 'Team'}, inplace=True)

#clean data
all_data['SuccessfulPlay'] = all_data['IsTouchdown'] | (~all_data['IsInterception'] & ~all_data['IsFumble'])
avg_conceded_plays_home = all_data.groupby('Home')['SuccessfulPlay'].mean()
avg_conceded_plays_visitor = all_data.groupby('Visitor')['SuccessfulPlay'].mean()
overall_avg_conceded_plays = (avg_conceded_plays_home + avg_conceded_plays_visitor) / 2
all_data['Turnover'] = all_data['IsInterception'] | all_data['IsFumble']
avg_forced_turnovers_home = all_data.groupby('Home')['Turnover'].mean()
avg_forced_turnovers_visitor = all_data.groupby('Visitor')['Turnover'].mean()
overall_avg_forced_turnovers = (avg_forced_turnovers_home + avg_forced_turnovers_visitor) / 2

#define data frame for team defensive statistics
team_features_defensive = pd.DataFrame({
    'Team': team_features['Team'].values,
    'AvgPointsDefended': team_features['AvgPointsAllowed'].values,
    'AvgConcededPlays': overall_avg_conceded_plays.values,
    'AvgForcedTurnovers': overall_avg_forced_turnovers.values
})

team_features_combined = team_features.merge(team_features_defensive, on='Team')

#advanced offensive stats
avg_yards_per_play_home = all_data.groupby('Home')['Yards'].mean()
avg_yards_per_play_visitor = all_data.groupby('Visitor')['Yards'].mean()
overall_avg_yards_per_play = (avg_yards_per_play_home + avg_yards_per_play_visitor) / 2
total_yards_per_game_home = all_data.groupby(['SeasonYear', 'Home'])['Yards'].sum() / all_data.groupby(['SeasonYear', 'Home']).size()
total_yards_per_game_visitor = all_data.groupby(['SeasonYear', 'Visitor'])['Yards'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()
overall_avg_yards_per_game = (total_yards_per_game_home + total_yards_per_game_visitor).groupby(level=1).mean()
avg_pass_completion_rate_home = all_data.groupby('Home').apply(lambda x: 1 - x['IsIncomplete'].mean())
avg_pass_completion_rate_visitor = all_data.groupby('Visitor').apply(lambda x: 1 - x['IsIncomplete'].mean())
overall_avg_pass_completion_rate = (avg_pass_completion_rate_home + avg_pass_completion_rate_visitor) / 2
avg_touchdowns_per_game_home = all_data.groupby(['SeasonYear', 'Home'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Home']).size()
avg_touchdowns_per_game_visitor = all_data.groupby(['SeasonYear', 'Visitor'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()
overall_avg_touchdowns_per_game = (avg_touchdowns_per_game_home + avg_touchdowns_per_game_visitor).groupby(level=1).mean()
avg_rush_success_rate_home = all_data.groupby('Home').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean())
avg_rush_success_rate_visitor = all_data.groupby('Visitor').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean())
overall_avg_rush_success_rate = (avg_rush_success_rate_home + avg_rush_success_rate_visitor) / 2

#define data frame for offensive statistics
new_offensive_features = pd.DataFrame({
    'Team': team_features_combined['Team'],
    'AvgYardsPerPlay': overall_avg_yards_per_play.values,
    'AvgYardsPerGame': overall_avg_yards_per_game.values,
    'AvgPassCompletionRate': overall_avg_pass_completion_rate.values,
    'AvgTouchdownsPerGame': overall_avg_touchdowns_per_game.values,
    'AvgRushSuccessRate': overall_avg_rush_success_rate.values
})

team_features_expanded = team_features_combined.merge(new_offensive_features, on='Team')

#advanced defensive stats
avg_yards_allowed_per_play_home = all_data.groupby('Home')['Yards'].mean()
avg_yards_allowed_per_play_visitor = all_data.groupby('Visitor')['Yards'].mean()
overall_avg_yards_allowed_per_play = (avg_yards_allowed_per_play_home + avg_yards_allowed_per_play_visitor) / 2
total_yards_allowed_per_game_home = all_data.groupby(['SeasonYear', 'Home'])['Yards'].sum() / all_data.groupby(['SeasonYear', 'Home']).size()
total_yards_allowed_per_game_visitor = all_data.groupby(['SeasonYear', 'Visitor'])['Yards'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()
overall_avg_yards_allowed_per_game = (total_yards_allowed_per_game_home + total_yards_allowed_per_game_visitor).groupby(level=1).mean()
avg_pass_completion_allowed_rate_home = all_data.groupby('Home').apply(lambda x: 1 - x['IsIncomplete'].mean())
avg_pass_completion_allowed_rate_visitor = all_data.groupby('Visitor').apply(lambda x: 1 - x['IsIncomplete'].mean())
overall_avg_pass_completion_allowed_rate = (avg_pass_completion_allowed_rate_home + avg_pass_completion_allowed_rate_visitor) / 2
avg_touchdowns_allowed_per_game_home = all_data.groupby(['SeasonYear', 'Home'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Home']).size()
avg_touchdowns_allowed_per_game_visitor = all_data.groupby(['SeasonYear', 'Visitor'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()
overall_avg_touchdowns_allowed_per_game = (avg_touchdowns_allowed_per_game_home + avg_touchdowns_allowed_per_game_visitor).groupby(level=1).mean()
avg_rush_success_allowed_rate_home = all_data.groupby('Home').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean())
avg_rush_success_allowed_rate_visitor = all_data.groupby('Visitor').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean())
overall_avg_rush_success_allowed_rate = (avg_rush_success_allowed_rate_home + avg_rush_success_allowed_rate_visitor) / 2

#define data frame for advanced defensive statistics
new_defensive_features = pd.DataFrame({
    'Team': team_features_expanded['Team'],
    'AvgYardsAllowedPerPlay': overall_avg_yards_allowed_per_play.values,
    'AvgYardsAllowedPerGame': overall_avg_yards_allowed_per_game.values,
    'AvgPassCompletionAllowedRate': overall_avg_pass_completion_allowed_rate.values,
    'AvgTouchdownsAllowedPerGame': overall_avg_touchdowns_allowed_per_game.values,
    'AvgRushSuccessAllowedRate': overall_avg_rush_success_allowed_rate.values
})

team_features_complete = team_features_expanded.merge(new_defensive_features, on='Team')

#get games to predict
upcomingGames = pd.read_csv("./Data/BEST New_Upcoming_Schedule (1).csv")
upcoming_encoded_home = upcomingGames.merge(team_features_complete, left_on='Home', right_on='Team', how='left')
upcoming_encoded_both = upcoming_encoded_home.merge(team_features_complete, left_on='Visitor', right_on='Team', suffixes=('_Home', '_Visitor'), how='left')
for col in ['AvgPointsScored', 'AvgPointsAllowed', 'WinRate', 'AvgPointsDefended', 'AvgConcededPlays', 'AvgForcedTurnovers',
            'AvgYardsPerPlay', 'AvgYardsPerGame', 'AvgPassCompletionRate', 'AvgTouchdownsPerGame', 'AvgRushSuccessRate',
            'AvgYardsAllowedPerPlay', 'AvgYardsAllowedPerGame', 'AvgPassCompletionAllowedRate', 'AvgTouchdownsAllowedPerGame', 'AvgRushSuccessAllowedRate']:
    upcoming_encoded_both[f'Diff_{col}'] = upcoming_encoded_both[f'{col}_Home'] - upcoming_encoded_both[f'{col}_Visitor']

upcoming_encoded_final = upcoming_encoded_both[['Home', 'Visitor'] + [col for col in upcoming_encoded_both.columns if 'Diff_' in col]]

#data to train
training_encoded_home = all_data.merge(team_features_complete, left_on='Home', right_on='Team', how='left')
training_encoded_both = training_encoded_home.merge(team_features_complete, left_on='Visitor', right_on='Team', suffixes=('_Home', '_Visitor'), how='left')
for col in ['AvgPointsScored', 'AvgPointsAllowed', 'WinRate', 'AvgPointsDefended', 'AvgConcededPlays', 'AvgForcedTurnovers',
            'AvgYardsPerPlay', 'AvgYardsPerGame', 'AvgPassCompletionRate', 'AvgTouchdownsPerGame', 'AvgRushSuccessRate',
            'AvgYardsAllowedPerPlay', 'AvgYardsAllowedPerGame', 'AvgPassCompletionAllowedRate', 'AvgTouchdownsAllowedPerGame', 'AvgRushSuccessAllowedRate']:
    training_encoded_both[f'Diff_{col}'] = training_encoded_both[f'{col}_Home'] - training_encoded_both[f'{col}_Visitor']
training_data = training_encoded_both[[col for col in training_encoded_both.columns if 'Diff_' in col]]
training_labels = training_encoded_both['HomeWon']

#regression model init
logreg = LogisticRegression(max_iter=1000)
cross_val_scores = cross_val_score(logreg, training_data, training_labels, cv=5)

#further clean data
nan_columns = training_data.columns[training_data.isna().any()].tolist()
nan_counts = training_data[nan_columns].isna().sum()
training_data_cleaned = training_data.dropna()
training_labels_cleaned = training_labels.loc[training_data_cleaned.index]
cross_val_scores_cleaned = cross_val_score(logreg, training_data_cleaned, training_labels_cleaned, cv=5)
cross_val_scores_cleaned_mean = cross_val_scores_cleaned.mean()

#regression model train
logreg.fit(training_data_cleaned, training_labels_cleaned)
upcoming_game_probabilities = logreg.predict_proba(upcoming_encoded_final[[col for col in upcoming_encoded_final.columns if 'Diff_' in col]])
upcoming_game_prob_home_win = upcoming_game_probabilities[:, 1]
upcoming_encoded_final['HomeWinProbability'] = upcoming_game_prob_home_win
upcoming_predictions = upcoming_encoded_final[['Home', 'Visitor', 'HomeWinProbability']].sort_values(by='HomeWinProbability', ascending=False)
