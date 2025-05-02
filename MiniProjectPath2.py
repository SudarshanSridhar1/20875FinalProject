import pandas 
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''

# 3 analysis questions
# 1 : Which bridges should have sensors to get best prediction of overall traffic? 3/4 can be used.
# 2: Is it possible to use next day weather forcast (low/high temp and precip) to predict total number of bicyclists?
# 3: Are there patterns associated with specific days of the week, and can you use the this data to predict what day today is based on the number of byciclys on bridge

dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
#print(dataset_2.to_string()) #This line will print out your data

# first step is to load and clean the data
csv = "nyc_bicycle_counts_2016.csv"
def load():
    data = pandas.read_csv(csv)
    bridges = ["Brooklyn Bridge", "Manhattan Bridge", "Queensboro Bridge", "Williamsburg Bridge", "Total"]
    for i in bridges:
        data[i] = pandas.to_numeric(data[i].str.replace(",",""), errors="coerce")
        # removes commas and converts into int
    return data.dropna() # removes missing values

def r2(ytrue, ypred):
    sse = ((ytrue - ypred) ** 2).sum()
    sst = ((ytrue - ytrue.mean()) ** 2).sum()
    if (sst != 0):
        return (1 - sse / sst)
    else:
        return 0
def kfoldr2(model, X, y, k):
    # average r2 over k folds using KFOLD from scikit learn
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    scores = []
    # loops over each fold manually
    for train, test in kf.split(X):
        Xtrain, Xtest =  X.iloc[train], X.iloc[test]
        ytrain, ytest = y.iloc[train], y.iloc[test]
        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)
        scores.append(r2(ytest.to_numpy(),ypred))
    mean = np.mean(scores)
    return float(mean)

def best(data):
    bridges = ["Brooklyn Bridge", "Manhattan Bridge", "Queensboro Bridge", "Williamsburg Bridge"]
    y = data["Total"] # what we trying to explain
    bestr = -1e9 # lowest possible
    bestset = None 
    for combo in itertools.combinations(bridges,3): #every way to pick 3 out of 4
        X = data[list(combo)]
        r2 = kfoldr2(LinearRegression(), X, y, k=5) # basic linear is fine for this
        if r2 > bestr:
            bestr, bestset = r2, combo
    print("Best Three Bridges:", bestset)
    print("Mean 5-fold r2 :", round(bestr, 3))
def plot(data):
    # bar chart of average riders for each day
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    data["Day"] = pandas.Categorical(data["Day"], categories= days, ordered= True)
    # makes sure in proper order and not alphabetical
    mean = data.groupby("Day")[["Brooklyn Bridge", "Manhattan Bridge", "Queensboro Bridge", "Williamsburg Bridge", "Total"]].mean()
    mean.plot(kind = "bar", figsize = (12,6))
    plt.ylabel("Average riders (2016)")
    plt.title("Average daily riders by weekday")
    plt.show()
def main():
    data = load()
    best(data)
    plot(data)


if __name__ == "__main__":
    main()



    
