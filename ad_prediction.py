'''

Hi, O'Reilly Media team! I genuinely enjoyed working through this project, and I'm thoroughly
eager to help unravel data using state-of-the art ML + optimization techniques.

Here's an outline of my Python script:
    1. Cleaning dataset
    2. Visualizing dataset
    3. Cross Validation among 3 ML classifiers
    4. Model Selection
    5. Predicting probability of ad clicked

Kindly,
Kitu Komya
    
'''

###############################################################################

'''
1. Cleaning dataset 
   Goal is to prepare dataset for ML classifier by factorizing variables and expanding the date-time variable.
'''

import pandas as pd

# reading in data
sampled_test = pd.read_csv("sampled_test.csv", header = None)
sampled_training = pd.read_csv("sampled_training.csv", header = None)

# rename headers
sampled_training.columns = ['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id',
                            'site_domain', 'site_category', 'app_id', 'app_domain',
                            'app_category', 'device_id', 'device_ip',
                            'device_model', 'device_type', 'device_conn_type', 
                            'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

sampled_test.columns = ['id', 'hour', 'C1', 'banner_pos', 'site_id',
                            'site_domain', 'site_category', 'app_id', 'app_domain',
                            'app_category', 'device_id', 'device_ip',
                            'device_model', 'device_type', 'device_conn_type', 
                            'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

# loop through both datasets to simplify data cleaning
datasets = [sampled_training, sampled_test]

for sets in datasets:
    # extract day and hour by first reassigning to datetime variable
    sets['hour'] = pd.to_datetime(sets['hour'], format = '%y%m%d%H')
    sets['real_hour'] = sets['hour'].dt.hour
    sets['weekday'] = sets['hour'].dt.day_name()
    del sets['hour']

    # encode variables to category by refactorizing to a distinct number
    col_names = list(sets) # create list of column names
    col_names.remove('id') # keep id unchanged
    for col in col_names: # loop through columns to change variable type
        sets[col] = pd.factorize(sets[col])[0]
    
    # ensure all data types are in good form
    print(sets.dtypes)

# delete unnecessary data  
del datasets, sets

###############################################################################

'''
2. Visualizing dataset
   Goal is to understand what our data looks like and to glean some intuition on how the classifier may work.
'''

# explore click-rate
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x = "click", data = sampled_training)
plt.show() # most of the times ads are not clicked!

# look into proportions...an ad is clicked on only 17% of the time. 
# should use f1-score instead of accuracy to measure success since imbalanced classes
sampled_training['click'].value_counts()/len(sampled_training)

###############################################################################

'''
3. Cross Validation
   Goal is to use cross-validation to quantify the f1-score of each ML classifier in order to select one.
'''

# import models
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# create training data without id or predicting variable
train_me = sampled_training.drop(['id', 'click'], axis = 1)

# choose models whose assumptions are met by our (transformed) categorical + numerical dataset
model_list = []
model_list.append(("Multinomial Naive Bayes", MultinomialNB()))
model_list.append(("Decision Tree", tree.DecisionTreeClassifier(criterion = "gini")))
model_list.append(("Logistic Regression", LogisticRegression()))

# evaluate each model's f1-score
results = []
names = []
for name, model in model_list:
	cv_results = cross_val_score(model, train_me, sampled_training['click'], cv = 5, scoring = "f1_macro")
	results.append(cv_results)
	names.append(name)

###############################################################################
    
'''
4. Model Selection
   Goal is to choose an ML classifier that works best on our dataset by comparing different algorithms.
'''

# compare algorithms' f1-score visually via boxplot distributions
fig = plt.figure()
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show() # these numbers are not bad in comparision to the baseline 17%

# let's use tree! explainable, meets our assumptions, and highest/consistent f1-score
model = tree.DecisionTreeClassifier(criterion = "gini")

# split data into 80/20 to verify model selection
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_me, sampled_training['click'], 
                                                    random_state = 0, test_size = 0.2)

# fit model on training data
model.fit(x_train, y_train)

# predict if ad will be clicked or not
y_pred = model.predict(x_test)
    
# measure precision and f1 score to ensure satisfiable
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred)) # looks good! let's use on our real testing set

###############################################################################

'''
5. Predicting probability of ad clicked
   Goal is to calculate the probability that an ad will be clicked using our chosen ML classifier (decision tree).
'''

# new dataframe to store id and probability score
prob_of_ad = pd.DataFrame(columns = ['id', 'prob_of_click'])
prob_of_ad['id'] = sampled_test['id']

# remove id since it's not a feature
sampled_test = sampled_test.drop('id', axis = 1)

# fit model and store probability of ad being clicked on testing data into list
prob = []
for row in sampled_test.itertuples():
    #print(model.predict_proba(sampled_test)[row.Index][1]) # to check values since predict_proba is a very slow function
    prob.append(model.predict_proba(sampled_test)[row.Index][1])

# append list to dataframe
prob_of_ad['prob_of_click'] = pd.Series(prob).values

# verify that algorithm is somewhat not non-sensical (should be near 0.17)
prob_of_ad['prob_of_click'].mean() # obtained 0.236! wow! not bad 

# write dataframe to csv file
prob_of_ad.to_csv("predictions.csv") 

# we see that the probability values are categorized (0, 0.333, 0.5, 0.666, 1, etc), 
# which makes sense, given our decision tree implementation