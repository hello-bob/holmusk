"""Costs of healthcare"""

"""Deliver value-based care with improved outcomes at reduced cost, through the use of analytics and decision support, 
including patient reported outcomes and medication adherence. The more efficiently resources are allocated in the 
healthcare system, the more patients are able to access these resources, including physicians, equipment, supplies,
and time. """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import pytz

"""Insert working directory here"""
os.getcwd()
file_path = '/Users/hydraze/Google Drive/Applications/Data science/Research/Holmusk/Task/datasciencepositionatholmusk'
os.chdir(file_path)

#creating dictionary to define data types when loading into pandas df
data_types = {'gender':'category',
              'race':'category',
              'resident_status':'category',
              'medical_history_1':'category',
              'medical_history_2':'category',
              'medical_history_3':'category',
              'medical_history_4':'category',
              'medical_history_5':'category',
              'medical_history_6':'category',
              'medical_history_7':'category',
              'preop_medication_1':'category',
              'preop_medication_2':'category',
              'preop_medication_3':'category',
              'preop_medication_4':'category',
              'preop_medication_5':'category',
              'preop_medication_6':'category',
              'symptom_1':'category',
              'symptom_2':'category',
              'symptom_3':'category',
              'symptom_4':'category',
              'symptom_5':'category',
              }
#Converting variables into categories will help with finding wrongly keyed data too


"""Loading dataset"""
bill_amount = pd.read_csv('./data/bill_amount.csv', dtype = data_types)
ids = pd.read_csv('./data/bill_id.csv', dtype = data_types, parse_dates= ['date_of_admission'])
clinical_data = pd.read_csv('./data/clinical_data.csv', dtype = data_types, parse_dates= ['date_of_admission', 'date_of_discharge'])
demographics = pd.read_csv('./data/demographics.csv', dtype = data_types, parse_dates= ['date_of_birth'])
#3000 assumed to be baseline because there should not be multiple entries in demographics

"""Checking for duplicates"""
demographics.drop_duplicates(inplace = True)
clinical_data.drop_duplicates(inplace = True)
ids.drop_duplicates(inplace = True)
bill_amount.drop_duplicates(inplace = True)
#No duplicates!

"""Checking out if ids are unique, because merging based on ids"""
#patient_id
print("Number of unique patient IDs in bill ids dataset = ",len(ids.patient_id.unique()))
print("Number of unique patient IDs in clinical data dataset = ",len(clinical_data.id.unique()))
print("Number of unique patient IDs in demographics dataset = ",len(demographics.patient_id.unique()))
#all datasets have 3000 unique patient ids

#bill
print("Number of unique bill IDs in bill ids dataset = ", len(ids.bill_id.unique()))
print("Number of unique bill IDs in bill amount dataset = ", len(bill_amount.bill_id.unique()))
#Both datasets have 13600 unique bills

"""Merging datasets"""
df_patients = demographics.merge(clinical_data, how = 'left', left_on = 'patient_id', right_on = 'id') #tested with inner & outer join first
df_patients.pop('id')
testing = df_patients.groupby('patient_id').count() #identifying counts of data per patient_id

df_bills = ids.merge(bill_amount, how = 'left', on = 'bill_id') #tested with inner & outer join first
testing = df_bills.groupby('bill_id').count().max() #identifying counts of data per bill_id

df_full = df_patients.merge(df_bills, how = 'left', on = ['patient_id', 'date_of_admission']) #tested with inner & outer join first
print(demographics.patient_id) #picked last patient_id for testing
testing = df_full.loc[df_full.patient_id == "20b609609d4dbb834d722ddf29f18879", :] #testing to see if left join was done correctly
print(len(df_full.patient_id.unique())) #3000 patients
print(len(df_full.bill_id.unique())) #13600 bills
col_names = df_full.columns

df_full.drop('bill_id', axis = 1, inplace = True) #Because bill_id is not needed anymore


"""Checking description of all variables"""
description = df_full.describe(include = 'all')
#some patient id appearing 16 times, 4 gender, 6 races, 4 resident status, medical history 3 has 4 cats,
#min bill amount as 79, and max as 81849. Should be good to look at the distribution. Data ranges from 2011-2016.
#missing values present in medical_history_2 and medical_history_5


"""Clearing up the variables with wrong spelling"""
#4 different genders
df_full.gender.unique() #shows Female, Male, f, m
df_full.loc[df_full.gender == 'f', 'gender'] = 'Female'
df_full.loc[df_full.gender == 'm', 'gender'] = 'Male'
df_full.gender.unique()
df_full.gender.cat.remove_unused_categories(inplace = True)
print(df_full.gender.cat.categories) #check unused categories removed
print([x for x in df_full.gender]) #check cleaning done correctly

#6 different races
df_full.race.unique()
df_full.loc[df_full.race == 'chinese', 'race'] = 'Chinese'
df_full.loc[df_full.race == 'India', 'race'] = 'Indian'
df_full.race.unique()
df_full.race.cat.remove_unused_categories(inplace = True)
df_full.race.cat.categories
print([x for x in df_full.race]) #check cleaning done correctly

#4 residential status
df_full.resident_status.unique()
df_full.loc[df_full.resident_status == 'Singapore citizen', 'resident_status'] = 'Singaporean'
df_full.resident_status.cat.remove_unused_categories(inplace = True)
print(df_full.resident_status.cat.categories)
print([x for x in df_full.resident_status]) #check cleaning done correctly

#4 categories in medical history 3
df_full.medical_history_3.value_counts() #0 is likely to indicate absence of medical condition, as it is way higher than
df_full.loc[df_full.medical_history_3 == 'No', 'medical_history_3'] = '0'
df_full.loc[df_full.medical_history_3 == 'Yes', 'medical_history_3'] = '1'
df_full.medical_history_3.cat.remove_unused_categories(inplace = True)
print(df_full.medical_history_3.cat.categories)
print([x for x in df_full.medical_history_3]) #check cleaning done correctly

"""Checking Missing values in all variables"""
df_full.isna().sum() #medical_history_5 has 1216 missing values, medical_history_2 has 932 missing values.
na_df = df_full.loc[((df_full.medical_history_2.isna()) | (df_full.medical_history_5.isna())), :]
print("Number of patients with NA in medical history 2 or 5 =", len(na_df.patient_id.unique())) #494 patients!!! that's approx 13% of all patients
na_df = df_full.loc[df_full.medical_history_5.isna(), :]
print("Number of patients with NA in medical history 5 =", len(na_df.patient_id.unique())) #301 patients with na in medical_history_5
na_df = df_full.loc[df_full.medical_history_2.isna(), :]
print("Number of patients with NA in medical history 2 =", len(na_df.patient_id.unique())) #233 patients with na in medical_history_2

#checking descriptions of those with NAs, and those without NAs
na_describe = na_df.describe(include = 'all')
not_na_df = df_full[((df_full.medical_history_2.notna()) & df_full.medical_history_5.notna())]
not_na_describe = not_na_df.describe(include = 'all')
#Comparing between variables of those with NAs and those who do not have NAs was not conclusive, because they seemed to share
#similar proportions and distributions for all variables. Carry on to impute missing values with 0s because by definition,
#because of two reasons. 1) 0 is the most likely outcome, and 2) NAs tend to be synonymous with 0

#Change NAs to 0, which is the median for both variables
df_full[['medical_history_2', 'medical_history_5']] = df_full[['medical_history_2', 'medical_history_5']].astype(np.number)
df_full.loc[df_full.medical_history_2.isna(), 'medical_history_2'] = 0
df_full.loc[df_full.medical_history_5.isna(), 'medical_history_5'] = 0
df_full.isna().any() #Checking if any of the columns have na values
df_full[['medical_history_2', 'medical_history_5']] = df_full[['medical_history_2', 'medical_history_5']].astype('category')
print(df_full.medical_history_2.cat.categories)

"""Identifying anomalous variables"""
#Checking if dates are in order
print("Number of birthdates after date of admission =", len(df_full.loc[df_full.date_of_birth > df_full.date_of_admission, :])) #No one's D.O.B is after date of admission
print("Number of date of admission after date of discharge =", len(df_full.loc[df_full.date_of_admission > df_full.date_of_discharge, :])) #No one's date of discharge is date of admission

"""Identifying if there are any patients not on ops medication"""
ops = df_full[['preop_medication_1', 'preop_medication_2', 'preop_medication_3', 'preop_medication_4', 'preop_medication_5', 'preop_medication_6']].copy()
ops['total'] = ops.astype(np.number).sum(axis = 1)
print("Max number of operation medication of all patients =", ops.total.max())
print("Min number of operation medication of all patients =", ops.total.min())
ops.loc[ops.total == 0, :] #8 entries without any surgical medication
no_ops = df_full.loc[ops.total == 0,:]
print("Number of patients not taking any preop medication =", len(no_ops.patient_id.unique()))

"""
In terms of medications, prior to surgery, an antibiotic may be given to prevent infections at the surgical site. 
Antibiotics are a category of drugs used to combat bacteria, and they are generally given orally (in pill form), or 
intravenously (through an IV) -- verywellhealth.com
More here: https://www.uclahealth.org/anes/what-medications-should-patients-take-before-surgery
"""


"""Feature engineering: Round 1"""
#Creation of age variable
df_full['year_of_birth'] = df_full.date_of_birth.map(
    lambda x: x.year
)

df_full['year_of_admission'] = df_full.date_of_admission.map(
    lambda x: x.year
)

df_full['age'] = 2020 - df_full.year_of_birth #Birthday in 2020. But might be inaccurate, because what if patients died by 2020.
print("Max age =", df_full.age.max(), ";" , "Min age =", df_full.age.min())
df_full['age_at_admission'] = df_full.year_of_admission - df_full.year_of_birth #Will determine later which age to use
print("Max age at admission =", df_full.age_at_admission.max(), ";", "Min age at admission =", df_full.age_at_admission.min())

#Creating days of admission variables
df_full['days_admitted'] = (df_full.date_of_discharge - df_full.date_of_admission).dt.days
print("Max days admitted =", df_full.days_admitted.max(), ";", "Min days admitted =", df_full.days_admitted.min())

#Creating BMI variable
df_full['bmi'] = df_full.weight/((df_full.height/100)**2)
print("Max bmi =", df_full.bmi.max(), ";", "Min bmi =", df_full.bmi.min())

#Number of bills per admission per patient id
num_bills = df_full[['patient_id', 'date_of_admission', 'amount']].groupby(['patient_id','date_of_admission'], as_index = False).count()
#This is to group bills by patient id, followed by date of admission. here, can count how many bill amounts there are per date of admission
print("Max number of bills =", num_bills.amount.max(), ";", "Min number of bills =", num_bills.amount.min())
#from this, can see that each patient for each admission has 4 bills. Will need to sum amount per admission

num_bills  = num_bills[['patient_id', 'date_of_admission']].groupby('patient_id', as_index = False).count()
#Got back 3000 patients again. On the right track.
num_bills.columns = ['patient_id', 'num_of_admissions'] #need to rename column now so that I don't have to rename later.
print("Max number of admissions =", num_bills.num_of_admissions.max(), ";", "Min number of admissions =", num_bills.num_of_admissions.min())
df_full = df_full.merge(num_bills, how = 'left', on = 'patient_id') #This will tag number of admissions to each participant
print(df_full.num_of_admissions.value_counts()) #Counts per number of admission.
df_full.num_of_admissions = df_full.num_of_admissions.astype('category')
#convert to categorical variable because more reflective. Also, effect on cost is not likely to be linear
print(num_bills.patient_id)
#df_full.loc[df_full.patient_id == "ffac3c4b6838f42625e1dd00dd7c867b", :] #check to see if the merge was correct


#Aggregate sum per admission and discharge. Also need to keep in mind of the people with multiple admissions
bills_per_admission = df_full[['patient_id', 'date_of_admission', 'amount']]
bills_per_admission = bills_per_admission.groupby(['patient_id','date_of_admission'], as_index = False).sum()
#Summing total bills per patient id per admission.
bills_per_admission.columns = ['patient_id', 'date_of_admission', 'amt_per_admission']
df_full = df_full.merge(bills_per_admission, how = 'left', on = ['patient_id','date_of_admission'])
#Same strategy as previous. Merge total cost to each patient id and date of admission
df_full.pop('amount') #don't need this anymore
df_full.drop_duplicates(inplace = True) #checking for duplicates/error in merging


"""Final dataset for data visualisation"""
df_full.columns
df_full = df_full[['patient_id', 'gender', 'race', 'resident_status', 'weight', 'height', 'bmi', 'age',
                   'age_at_admission', 'num_of_admissions', 'days_admitted', 'medical_history_1', 'medical_history_2',
                   'medical_history_3', 'medical_history_4',
                   'medical_history_5', 'medical_history_6', 'medical_history_7','preop_medication_1',
                   'preop_medication_2', 'preop_medication_3', 'preop_medication_4', 'preop_medication_5',
                   'preop_medication_6', 'symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5', 'lab_result_1',
                   'lab_result_2', 'lab_result_3', 'amt_per_admission']] #Just to reorganise the columns into demographics and clinical info

#'date_of_birth', 'date_of_admission', 'date_of_discharge' not included anymore because age will contain information from
#those variables

#Description of additional variables
description = df_full.describe(include = 'all') #Descriptive statistics of all columns. Just to get a feel of the data.
"""END OF PRE-PROCESSING. MOVE ON TO DATA VISUALISATION"""


"""Import additional packages"""
import sklearn
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split, ParameterGrid, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import ElasticNet
import math
from yellowbrick.regressor import ResidualsPlot, AlphaSelection, CooksDistance, PredictionError


"""Need label encoding for ordinal categorical variables, dummy encoding for nominal categorical variables.
There is no need to do onehotencoding because this is to identify factors rather than to develop a model which is capable
of predicting even outliers. Rmb this rationale"""
df_full.reset_index(inplace = True, drop = True) #Resetting index because they are still retained from previous steps
df_full.drop('patient_id', axis = 1, inplace = True) #Won't need this anymore
df_full.num_of_admissions = df_full.num_of_admissions.astype(np.int) #conversion to float because ordinal category
dummies = pd.get_dummies(df_full[['gender', 'race', 'resident_status']]) #These variables are nominal, so create dataframe for dummies
df_full = pd.concat([dummies, df_full.drop(['gender','race', 'resident_status'], axis = 1)], axis = 1)
#then concat df_full to dummies, while removing the original variables which have been dummy-fied


"""Split into training set and test set"""
seed = 2604

X = df_full.iloc[:,:-1] #separating predictor variables and outcome variable
Y = df_full.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = seed, shuffle = True)
#No need validation set because will use K-Fold cross validation later on to optimise model

"""Random Forest"""
#Here, I will find the best parameters for our current dataset first
#Also, take note to standardisation yet because RF does not require/perform better with standardisation
rf = RandomForestRegressor(random_state=seed)
rf_grid = {'max_features': ['auto'],
            'n_estimators': [500,700,900],
            'max_depth': [20],
            'min_samples_leaf': [1]
            }

#making root mean squared log error scorer because this overpenalises underprediction. when it comes to heathcare costs,
#it is always good to overpredict in my opinion. Will be used for the rest of the models
def rmsle(y_actual, y_predicted):
    rmsle = math.sqrt(mean_squared_log_error(y_actual, y_predicted))
    return rmsle

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

"""Setting K-Fold function to split the training set into training + validation sets (20%) when running RF later"""
kf = KFold(5, shuffle = True, random_state=0)

"""Grid search for best parameters for dataset"""
grid_search = GridSearchCV(rf, param_grid= rf_grid, scoring= rmsle_scorer, n_jobs = -1, cv = kf, refit = True)
grid_search.fit(x_train, y_train)
grid_search.best_params_ #max_depth: 20, min_samples_leaf:1, n_estimators = 700
-grid_search.best_score_
#grid_search.refit_time_ #14 seconds. Pretty fast


"""Using the best parameters for selection of variables. Will consider interpretability of model at this point in time.
Selection also based on RMSLE score + OOB score"""
predictor_cols = df_full.columns[:-1] #Defining the columns to include in the model

#Defining function to Input columns and automate outputs
def rf_generator(predictor_cols): #Has to be a list
    global rf #make rf into a global function so that can call again later for testing
    rf = RandomForestRegressor(n_estimators= 700,max_depth=20, min_samples_leaf=1, n_jobs = -1,
                               random_state= seed, oob_score=True, verbose = 1)
    scores = -cross_val_score(rf, x_train[predictor_cols], y_train, scoring= rmsle_scorer, n_jobs = -1, cv = kf)
    print("scores of cross-validation =", np.round(scores,3)) #[0.145 0.129 0.132 0.147 0.142]
    print("mean scores of cross-validation =", round(np.mean(scores), 3)) #0.139
    rf.fit(x_train[predictor_cols], y_train)
    print("Out-of-bag score = ", round(rf.oob_score_,3)) #0.9. Oob score as a metric of being able to predict what's in unseen data

    #This is for creating feature importance plot
    importances = pd.DataFrame(rf.feature_importances_).T
    importances.columns = df_full[predictor_cols].columns #rf.estimators will pull every single tree
    global indices #call as global first so can see all the indices later on
    indices = importances.sort_values(by = 0, axis = 1, ascending= False) #sorts from largest to smallest
    print(indices)

#plotting the feature importance plot
    sns.set_style('whitegrid')
    sns.set_palette(sns.color_palette("Blues_r"))
    plt.figure(figsize=(15,6))

    ax = sns.barplot(indices.columns, indices.iloc[0, :], palette = plt.cm.Blues(indices.iloc[0, :]*50))
    plt.title("Feature importances")
    plt.ylabel("Level of Importance")
    ax.set_xticklabels(rotation = 45, ha = 'right', labels = indices.columns)
    plt.tight_layout()
    plt.show()

"""from this part onwards, was the optimisation before adding in the new variables


rf_generator(predictor_cols)
#Here, it can be seen that symptom 5, being a foreigner, race_Malay are the top most contributers to the model. None of the
#pre-ops medication appeared to be significant. Same with gender. Will remove all variables which did not seem to be important
# (i.e. very close to 0)to see how it affects RMSLE score + OOB score.

print([x for x in indices])
cols_first = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age', 'age_at_admission', 'medical_history_1',
                  'weight', 'symptom_3', 'bmi', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
                  'resident_status_PR', 'symptom_4', 'lab_result_3', 'resident_status_Singaporean', 'symptom_1',
                  'lab_result_1', 'height', 'race_Indian', 'days_admitted']
rf_generator(cols_first)
#Improved OOB score to 0.902 and RMSLE to 0.137

Try to remove those overlapping variables e.g. weight, height, bmi??
without_height_and_weight = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age', 'age_at_admission', 'medical_history_1',
                  'symptom_3', 'bmi', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
                  'resident_status_PR', 'symptom_4', 'lab_result_3', 'resident_status_Singaporean', 'symptom_1',
                  'lab_result_1', 'race_Indian', 'days_admitted']
rf_generator(without_height_and_weight)
#It does better!! How about just including weight instead of bmi? because weight had higher importance than bmi
without_height_and_bmi = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age', 'age_at_admission', 'medical_history_1',
                  'symptom_3', 'weight', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
                  'resident_status_PR', 'symptom_4', 'lab_result_3', 'resident_status_Singaporean', 'symptom_1',
                  'lab_result_1', 'race_Indian', 'days_admitted']
rf_generator(without_height_and_bmi)

#Removing either age or age at admission
without_age = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
                  'symptom_3', 'weight', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
                  'resident_status_PR', 'symptom_4', 'lab_result_3', 'resident_status_Singaporean', 'symptom_1',
                  'lab_result_1', 'race_Indian', 'days_admitted']
rf_generator(without_age)

without_age_at_admission = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age', 'medical_history_1',
                  'symptom_3', 'weight', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
                  'resident_status_PR', 'symptom_4', 'lab_result_3', 'resident_status_Singaporean', 'symptom_1',
                  'lab_result_1', 'race_Indian', 'days_admitted']
rf_generator(without_age_at_admission)
#age at admission was better

#days admitted (bc it was engineered). Also, given that there are multiple rows of the same participant,
remove_days_admitted = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
                  'symptom_3', 'weight', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
                  'resident_status_PR', 'symptom_4', 'lab_result_3', 'resident_status_Singaporean', 'symptom_1',
                  'lab_result_1', 'race_Indian']
rf_generator(remove_days_admitted)

#Remove Resident status PR and Singaporean to implicitly indicate foreigner
pr_sg_removed = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
                  'symptom_3', 'weight', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
                  'symptom_4', 'lab_result_3','symptom_1',
                  'lab_result_1', 'race_Indian']
rf_generator(pr_sg_removed)
#Poor score, add both variables back in

#Removing last 2: race_indian and lab_result_1
remove_last_2 = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
                  'symptom_3', 'weight', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
                  'resident_status_PR', 'symptom_4', 'lab_result_3', 'resident_status_Singaporean', 'symptom_1']
rf_generator(remove_last_2)

#Removing last 2: 'resident_status_Singaporean' & 'symptom_1'
remove_last_2 = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
                  'symptom_3', 'weight', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
                  'resident_status_PR', 'symptom_4', 'lab_result_3']
rf_generator(remove_last_2)
#Score drops

remove_symptom_1 = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
                  'symptom_3', 'weight', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
                  'resident_status_PR', 'symptom_4', 'lab_result_3', 'resident_status_Singaporean']
rf_generator(remove_symptom_1)
#Score drops a lot

remove_singaporean = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
                  'symptom_3', 'weight', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
                  'resident_status_PR', 'symptom_4', 'lab_result_3', 'symptom_1']
rf_generator(remove_last_2)
#Similar to including both 'resident_status_Singaporean' & 'symptom_1'. Use the one with singaporean bc it can help with
#interpretability


Test on test set
best_cols = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
                  'symptom_3', 'weight', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
                  'resident_status_PR', 'symptom_4', 'lab_result_3', 'resident_status_Singaporean', 'symptom_1']
rf_generator(best_cols)

predict = rf.predict(x_test[best_cols])

print("rmsle of prediction using test set = ",round(rmsle(y_test, predict), 3))
#Because the test set gave a rmsle which is substantially lower than training set (0.126), there is still room for
#optimisation. Some data is not being used, which could help the model training improve

sns.set_style("whitegrid")
sns.scatterplot(y_test, predict)
plt.plot(range(0,90000),range(0,90000), "k-")
plt.title("Scatterplot of predicted bill amount against actual bill amount")
plt.xlabel("Actual bill amount")
plt.ylabel("Predicted bill amount")
"""

"""
Impurity-based feature importance can inflate the importance of numerical features. Seems like it is not the case in our
dataset though so it should be fine. 

Permutation importance analysis shows us which variables influence the ensemble model predictions on a global level,
potentially identifying more actionable targets. We found that LOS is by far the strongest predictor. Interventions
aimed at minimizing the effects of these variables can improve efficiency in the resource-limited healthcare system,
leading to higher quality care and improved outcomes for more patients.
"""

"""Feature engineering part 2!"""
df_full.columns
df_full["num_of_diseases"] = df_full[['medical_history_1', 'medical_history_2', 'medical_history_3', 'medical_history_4', 'medical_history_5', 'medical_history_6',
       'medical_history_7']].astype(np.number).sum(axis = 1)
print("Max number of medical history =", df_full.num_of_diseases.max(), ";", "Min number of medical history =", df_full.num_of_diseases.min())

df_full["num_of_preop_med"] = df_full[['preop_medication_1', 'preop_medication_2', 'preop_medication_3',
                                       'preop_medication_4', 'preop_medication_5',
                                       'preop_medication_6']].astype(np.number).sum(axis = 1)
print("Max number of preop meds =", df_full.num_of_preop_med.max(), ";", "Min number of preop meds =", df_full.num_of_preop_med.min())


df_full["num_of_symptoms"] = df_full[['symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5']].astype(np.number).sum(axis = 1)
print("Max number of symptoms =", df_full.num_of_symptoms.max(), ";", "Min number of symptoms =", df_full.num_of_symptoms.min())

df_full.columns

df_full = df_full[['gender_Female', 'gender_Male', 'race_Chinese', 'race_Indian',
       'race_Malay', 'race_Others', 'resident_status_Foreigner',
       'resident_status_PR', 'resident_status_Singaporean', 'weight', 'height',
       'bmi', 'age', 'age_at_admission', 'num_of_admissions', 'days_admitted',
       'medical_history_1', 'medical_history_2', 'medical_history_3',
       'medical_history_4', 'medical_history_5', 'medical_history_6',
       'medical_history_7', 'preop_medication_1', 'preop_medication_2',
       'preop_medication_3', 'preop_medication_4', 'preop_medication_5',
       'preop_medication_6', 'symptom_1', 'symptom_2', 'symptom_3',
       'symptom_4', 'symptom_5', 'lab_result_1', 'lab_result_2',
       'lab_result_3', 'num_of_diseases', 'num_of_preop_med', 'num_of_symptoms', 'amt_per_admission']]




"""Testing out new variables by adding them to best rf model so far"""
X = df_full.iloc[:,:-1] #separating predictor variables and outcome variable
Y = df_full.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = seed, shuffle = True)

with_new_var = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
            'symptom_3', 'weight', 'symptom_2', 'medical_history_6', 'race_Chinese', 'lab_result_2',
            'resident_status_PR', 'symptom_4', 'lab_result_3', 'resident_status_Singaporean', 'symptom_1',
             'num_of_diseases', 'num_of_preop_med', 'num_of_symptoms']
rf_generator(with_new_var)
#Vastly improved scores of oob score = 0.931, and mean rmsle = 0.104. symptoms lost its importance, except symptom 5

remove_symptoms = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
             'weight',  'medical_history_6', 'race_Chinese', 'lab_result_2',
            'resident_status_PR', 'lab_result_3', 'resident_status_Singaporean',
             'num_of_diseases', 'num_of_preop_med', 'num_of_symptoms']
rf_generator(remove_symptoms)
#Same scores. Keep model because smaller

"""From here on just removing the bottom few variables to see if it improves the score or not"""
remove_total_pre_ops = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
             'weight',  'medical_history_6', 'race_Chinese', 'lab_result_2',
            'resident_status_PR', 'lab_result_3', 'resident_status_Singaporean',
             'num_of_diseases', 'num_of_symptoms']
rf_generator(remove_total_pre_ops)
#Scores improved. RMSLE = 0.103; OOB score = 0.933

remove_singaporean = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
             'weight',  'medical_history_6', 'race_Chinese', 'lab_result_2',
            'resident_status_PR', 'lab_result_3', 'num_of_diseases', 'num_of_symptoms']
rf_generator(remove_singaporean)


remove_lab_result_3 = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
             'weight',  'medical_history_6', 'race_Chinese', 'lab_result_2',
            'resident_status_PR', 'num_of_diseases', 'num_of_symptoms']
rf_generator(remove_lab_result_3)

remove_pr = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
             'weight',  'medical_history_6', 'race_Chinese', 'lab_result_2', 'num_of_diseases', 'num_of_symptoms']
rf_generator(remove_pr)
#Stop here

"""Test on test set"""
best_cols = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
             'weight',  'medical_history_6', 'race_Chinese', 'lab_result_2',
            'resident_status_PR', 'num_of_diseases', 'num_of_symptoms']
rf_generator(best_cols)

predict = rf.predict(x_test[best_cols])

print("rmsle of prediction using test set = ",round(rmsle(y_test, predict), 3)) #0.092
    #Test set still gave rmsle which is much lower than there is still room for
    #optimisation. Some data is not being used, which could help the model training improve


sns.set_style("darkgrid")
sns.regplot(y_test, predict, scatter_kws={'alpha': 0.4})
plt.plot(range(0,90000),range(0,90000), "k-")
plt.annotate("R^2 =" + str(np.round(np.square((np.corrcoef(y_test, predict)[0][1])),3)), xy= (80000,10000))
plt.title("Regression plot of predicted bill amount against actual bill amount")
plt.xlabel("Actual bill amount")
plt.ylabel("Predicted bill amount")

print("The correlation coef =", np.round(np.corrcoef(y_test, predict)[0][1],3)) #corrcoef = 0.968



"""ElasticNet Regression: To see if linear model works. Also, using the best set of variables, engineer diff variants of 
continuous variables + standardising the x-test and y-test"""
best_df = df_full[['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'age_at_admission', 'medical_history_1',
             'weight',  'medical_history_6', 'race_Chinese', 'lab_result_2',
            'resident_status_PR', 'num_of_diseases', 'num_of_symptoms', 'amt_per_admission']]

#defining standardiser so it's easier to standardise variables later
def standardiser(col):
    global std_col
    std_col = (col-col.mean())/col.std()
    return(std_col)

"""
#Looking at distribution of certain manipulated variables
sns.distplot(standardiser(best_df.age_at_admission)) #not normal-like
sns.distplot(standardiser(best_df.weight))
sns.distplot(standardiser(best_df.lab_result_2))
sns.distplot(standardiser(best_df.num_of_symptoms))
sns.distplot(standardiser(best_df.num_of_diseases)) #not normal-like
sns.distplot(standardiser(best_df.amt_per_admission)) #not normal-like

#Trying other manipulation to the variables
sns.distplot(np.log1p(best_df.age_at_admission)) #more normal. Still have slight skew
sns.distplot(np.log1p(best_df.num_of_diseases)) #not normal-like
sns.distplot(np.log1p(best_df.amt_per_admission)) #better

sns.distplot(np.sqrt(best_df.age_at_admission)) #more normal. No skew
sns.distplot(np.sqrt(best_df.num_of_diseases)) #not normal-like
sns.distplot(np.sqrt(best_df.amt_per_admission)) #not so good. stick with log transform
"""

"""Don't transform number of diseases and number of symptoms. Sqrt transform age, log transform amt_per admission. 
Standardise weight and lab result 2 AFTER splitting test and training set"""
best_df["sqrt_age"] = np.sqrt(best_df.age_at_admission)
best_df["log_amt"] = np.log(best_df.amt_per_admission)
best_df = best_df[['symptom_5', 'resident_status_Foreigner', 'race_Malay',
       'age_at_admission', 'medical_history_1', 'weight', 'medical_history_6',
       'race_Chinese', 'lab_result_2', 'resident_status_PR', 'num_of_diseases',
       'num_of_symptoms', 'sqrt_age', 'log_amt']]

"""Split into training set and test set"""
seed = 2604

X = best_df.iloc[:,:-1] #separating predictor variables and outcome variable
Y = best_df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = seed, shuffle = True)

#Standardising weight and lab results 2
x_train["std_weight"] = standardiser(x_train.weight)
x_test["std_weight"] = standardiser(x_test.weight)
x_train["std_lab_result_2"] = standardiser(x_train.lab_result_2)
x_test["std_lab_result_2"] = standardiser(x_test.lab_result_2)


"""Optimisation of elastic net"""
elastic_grid = [{'alpha': [0.0005, 0.1, 0.5, 0.9, 0.95, 0.99, 1],
          'l1_ratio': np.arange(0,1+0.1,0.1)}]

elastic_reg = ElasticNet(random_state=seed, tol = 0.15) #Have to increase tolerance otherwise cannot converge

#Elastic Net function to test out the different parameters
x_train.columns
predictor_cols = ['symptom_5', 'resident_status_Foreigner', 'race_Malay',
       'age_at_admission', 'medical_history_1', 'weight', 'medical_history_6',
       'race_Chinese', 'lab_result_2', 'resident_status_PR', 'num_of_diseases',
       'num_of_symptoms', 'sqrt_age', 'std_weight', 'std_lab_result_2']


#Creation of elastic net function similar to random forest
def cv_elastic_net(predictor_cols):
    grid_search = GridSearchCV(elastic_reg, param_grid= elastic_grid, scoring= rmsle_scorer, n_jobs = -1, cv = kf,
                               refit = True)
    grid_search.fit(x_train[predictor_cols], y_train)
    print("Train score = ",grid_search.best_params_) #alpha = 0.0005, l1 ratio = 0.0. Hence, favouring ridge regression
    predict = grid_search.predict(x_test[predictor_cols])

    print("RMSLE of Elastic Net =", np.round(rmsle(y_test,predict),5))

    sns.set_style("whitegrid")
    plt.figure(figsize = (9,6))
    sns.scatterplot(y_test, predict)
    plt.plot((8.5,11.5),(8.5,11.5), "k-")
    plt.title("Scatterplot of predicted bill amount (logged) against actual bill amount (logged)")
    plt.xlabel("Actual bill amount (logged)")
    plt.ylabel("Predicted bill amount (logged)")

#No error only when logged

cv_elastic_net(predictor_cols) #testing function. Best params found to be alpha: 0.0005 and l1_ratio = 0.1 i.e. Ridge regression

#Using original variables
vanilla = ['symptom_5', 'resident_status_Foreigner', 'race_Malay',
       'age_at_admission', 'medical_history_1', 'weight', 'medical_history_6',
       'race_Chinese', 'lab_result_2', 'resident_status_PR', 'num_of_diseases',
       'num_of_symptoms']
cv_elastic_net(vanilla) #RMSLE of 0.00852. Best params still found to be alpha: 0.0005 and l1_ratio = 0.0

"""Defining the elastic_net function to collect training and test scores"""
def elastic_generator(predictor_cols): #Has to be a list
    global elastic  # make elastic into a global function so that can call again later for testing
    elastic = ElasticNet(alpha=0.0005, l1_ratio=0.05, random_state=seed, tol=0.15)
    scores = -cross_val_score(elastic, x_train[predictor_cols], y_train, scoring=rmsle_scorer, n_jobs=-1, cv=kf)
    print("scores of cross-validation =", np.round(scores, 3))
    print("mean scores of cross-validation =", round(np.mean(scores), 5))
    elastic.fit(x_train[predictor_cols], y_train)
    train_pred = elastic.predict(x_train[predictor_cols])  # training predictions
    print("rmsle of training =", np.round(rmsle(y_train, train_pred), 5))  # training prediction rmsle
    test_pred = elastic.predict(x_test[predictor_cols])
    print("rmsle of test =", np.round(rmsle(y_test, test_pred), 5))

    sns.set_style("whitegrid")
    plt.figure(figsize=(9, 6))
    sns.scatterplot(y_train, train_pred, label="Training set", color="darkblue", alpha=0.3)
    sns.scatterplot(y_test, test_pred, label="Test set", color="orange", alpha=0.3)
    plt.plot((8.5, 11.5), (8.5, 11.5), "k-")
    plt.annotate("R^2 =" + str(np.round(np.square((np.corrcoef(y_test, test_pred)[0][1])), 3)), xy=(11, 8.75))
    plt.legend()
    plt.title("Scatterplot of predicted bill amount against actual bill amount (Logged)")
    plt.xlabel("Actual bill amount (Logged)")
    plt.ylabel("Predicted bill amount (Logged)")

elastic_generator(vanilla) #rmsle of test = 0.00852; rmsle of training = 0.00758

print(predictor_cols)

"""Using newly derived variables instead of vanilla variables"""
test_cols = ['symptom_5', 'resident_status_Foreigner', 'race_Malay', 'medical_history_1',
             'medical_history_6', 'race_Chinese', 'resident_status_PR', 'num_of_diseases', 'num_of_symptoms',
             'sqrt_age', 'std_weight', 'std_lab_result_2']
elastic_generator(test_cols) #rmsle of test = 0.0084; rmsle of training = 0.00743

"""Simplifying model"""
"""Removing the weakest variables as seen from random forest using best_cols"""
print("Removing number of diseases")
remove_num_diseases = ['symptom_5', 'resident_status_Foreigner', 'race_Malay',
                       'age_at_admission', 'medical_history_1', 'weight', 'medical_history_6',
                       'race_Chinese', 'lab_result_2', 'resident_status_PR',
                       'num_of_symptoms']

elastic_generator(remove_num_diseases)
print("\n")

print("Removing lab results 2")
remove_lab2 = ['symptom_5', 'resident_status_Foreigner', 'race_Malay',
                   'age_at_admission', 'medical_history_1', 'weight', 'medical_history_6',
                   'race_Chinese', 'resident_status_PR',
                   'num_of_symptoms']

elastic_generator(remove_lab2)
print("\n")

print("Removing medical history 6")
remove_med6 = ['symptom_5', 'resident_status_Foreigner', 'race_Malay',
               'age_at_admission', 'medical_history_1', 'weight', 'race_Chinese', 'resident_status_PR',
               'num_of_symptoms']

elastic_generator(remove_med6)
print("\n")

print("Removing just lab results 2")
remove_lab2_only = ['symptom_5', 'resident_status_Foreigner', 'race_Malay',
                       'age_at_admission', 'medical_history_1', 'weight', 'medical_history_6',
                       'race_Chinese', 'num_of_diseases', 'resident_status_PR',
                       'num_of_symptoms']

elastic_generator(remove_lab2_only)
print("\n")

"""So I tried to reduce the side of the model further by removing the vanilla variables which were found to be the least 
important from Random Forest. I removed 3 variables of the least importance. I found out that the scores obtained from 
removing just number of diseases and from removing both number of diseases and lab result 2 were exactly the same. This 
means that lab result 2 is not as important as it seems. The 4th run was to just remove lab result 2. As seen above, it 
results in the same R^2 although the RMSLE scores was slightly poorer (increased). This means that we can remove lab_result_2, 
if we wanted a simpler model."""

"""END OF ANALYSIS"""

