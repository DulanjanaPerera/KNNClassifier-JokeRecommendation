import numpy as np
import pandas as pd
import pylab as plt
import random as rnd
import math
from KNNClassifier import KNNClassifier

joke_r = pd.read_csv("modified_jester_data.csv", header=None)
joke_r = np.array(joke_r)
jokes = pd.read_csv("jokes.csv", header=None)
jokes = np.array(jokes.iloc[:, 1])

tr_ut = joke_r[:900]
ts_ut = joke_r[900:]

knn = KNNClassifier()

''' Evaluate the total error of 100 users '''
# userError, totalerror = knn.eval_joker(tr_ut, 5, ts_ut)
# print(totalerror)

''' Analysis of K values for the system. Best K is used for the recommender system '''
# xaxis = np.array(range(5, 25, 5))
# mae_total = []
# for k in xaxis:
#     userError, totalerror = knn.eval_joker(tr_ut, k, ts_ut)
#     mae_total.append(totalerror)
#     print("total MAE : ", totalerror)
# nmae = np.array(mae_total)
# plt.plot(xaxis, nmae)
# plt.title("Comparing K values for KNN prediction - Joker")
# plt.xlabel("K")
# plt.ylabel("Total Mean Absolute Error")
# plt.minorticks_on()
# plt.grid(which='both', alpha=0.5)
# plt.show()

# Compute the recommendations
recoJokes, rating = knn.recommendations(tr_ut, ts_ut[87], jokes, 3, 5)
print("Recommended jokes: ")
for i in range(len(recoJokes)):
    print(recoJokes[i])
print(*rating, sep='\n')