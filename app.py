
from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
df = pd.read_csv('ipl_data.csv')

def create_visualizations():
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='bat_team')
    plt.title('Matches Played by Batting Team')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/bat_team_count.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='bowl_team')
    plt.title('Matches Played by Bowling Team')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/bowl_team_count.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    df['runs'].hist(bins=30)
    plt.title('Distribution of Runs')
    plt.savefig('static/runs_hist.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='bat_team', y='runs', data=df)
    plt.xticks(rotation=45)
    plt.title('Runs per Batting Team')
    plt.savefig('static/boxplot_bat_runs.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='overs', y='runs')
    plt.title('Runs vs Overs')
    plt.savefig('static/runs_vs_overs.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='YlGnBu')
    plt.title('Correlation Heatmap')
    plt.savefig('static/heatmap.png')
    plt.close()

create_visualizations()

model_df = df[['bat_team', 'bowl_team', 'overs', 'wickets', 'runs_last_5', 'wickets_last_5', 'runs']].copy()
model_df = pd.get_dummies(model_df, columns=['bat_team', 'bowl_team'])

X = model_df.drop('runs', axis=1)
y = model_df['runs']

model = LinearRegression()
model.fit(X, y)

model_columns = X.columns  

teams = sorted(df['bat_team'].unique())
players = sorted(df['batsman'].unique())

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        bat_team = request.form['bat_team']
        bowl_team = request.form['bowl_team']
        overs = float(request.form['overs'])
        wickets = int(request.form['wickets'])
        runs_last_5 = int(request.form['runs_last_5'])
        wickets_last_5 = int(request.form['wickets_last_5'])

        input_dict = {
            'overs': overs,
            'wickets': wickets,
            'runs_last_5': runs_last_5,
            'wickets_last_5': wickets_last_5
        }

       
        for team in df['bat_team'].unique():
            input_dict[f'bat_team_{team}'] = 1 if team == bat_team else 0

        for team in df['bowl_team'].unique():
            input_dict[f'bowl_team_{team}'] = 1 if team == bowl_team else 0

        input_df = pd.DataFrame([input_dict])

       
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]

    return render_template('index.html', teams=teams, players=players, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
