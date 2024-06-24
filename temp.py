import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

#Load the datasets
dfa = pd.read_excel("case_study1.xlsx")
dfb = pd.read_excel("case_study2.xlsx")

df1 = dfa.copy()
df2 = dfb.copy()

columns_need_to_remove_df1 = []

for i in df1.columns:
    if df1.loc[df1[i] == -99999].shape[0]>10000: 
        columns_need_to_remove_df1.append(i)
        
for i in df1.columns:
    df1 = df1.loc[df1[i] != -99999]
    
columns_need_to_remove_df2 = []

for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0]>10000: 
        columns_need_to_remove_df2.append(i)
        
df2 = df2.drop(columns_need_to_remove_df2, axis=1)

for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]
    
df = pd.merge(df1, df2, how="inner", left_on=["PROSPECTID"], right_on=["PROSPECTID"])

obj_cols = []
for i in df.columns:
    if df[i].dtype == "object":
        obj_cols.append(i)
 
# Chi square test
for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2','first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '---', pval)
    
numeric_columns = []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID', 'Approved_Flag']:
        numeric_columns.append(i)


# VIF sequentially check to remove multicolinearity
vif_data = df[numeric_columns]
total_columns= vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range(0, total_columns):
    vif_value = variance_inflation_factor(vif_data, column_index)
    print(column_index, '---', vif_value)                                                                                                                              
    
    if vif_value < 6:
        columns_to_be_kept.append(numeric_columns[i])
        column_index = column_index + 1
    else:
        vif_data = vif_data.drop([numeric_columns[i]], axis=1)
        
columns_to_be_kept_numerical = []
for i in columns_to_be_kept:
    a = list(df[i])
    b = list(df['Approved_Flag'])
    
    group_p1 = [value for value, group in zip(a, b) if group == 'P1']
    group_p2 = [value for value, group in zip(a, b) if group == 'P2']
    group_p3 = [value for value, group in zip(a, b) if group == 'P3']
    group_p4 = [value for value, group in zip(a, b) if group == 'P4']
    
    f_stats, p_value = f_oneway(group_p1, group_p2, group_p3, group_p4)
    if p_value <= 0.5:
        columns_to_be_kept_numerical.append(i)
        
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]

df['MARITALSTATUS'].unique()
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()


df.loc[df['EDUCATION'] == 'SSC', ['EDUCATION']] = 1
df.loc[df['EDUCATION'] == '12TH', ['EDUCATION']] = 2
df.loc[df['EDUCATION'] == 'GRADUATE', ['EDUCATION']] = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE', ['EDUCATION']] = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE', ['EDUCATION']] = 4
df.loc[df['EDUCATION'] == 'OTHERS', ['EDUCATION']] = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL', ['EDUCATION']] = 3

df['EDUCATION'].value_counts()
df['EDUCATION'] = df['EDUCATION'].astype(int)
df.info()


df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])
df.info()

final_acc = []

X = df_encoded.drop(['Approved_Flag'], axis=1)
y = df_encoded['Approved_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)


acc_score = accuracy_score(y_test, y_pred)
print()
print(f"Accuracy: {acc_score}")
print()
precision, recall, fl_score, _=precision_recall_fscore_support(y_test, y_pred) 


for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}: ")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {fl_score[i]}")
    print()
final_acc.append(f"Random forest acc score: {acc_score}")

#xgboost now
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

X = df_encoded.drop(['Approved_Flag'], axis=1)
y = df_encoded['Approved_Flag']

label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

xgb_classifier.fit(X_train, y_train)
y_pred = xgb_classifier.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
print()
print(f"Accuracy: {acc_score}")
print()
precision, recall, fl_score, _=precision_recall_fscore_support(y_test, y_pred) 


for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}: ")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {fl_score[i]}")
    print()

final_acc.append(f"xgb accuracy score is: {acc_score}")

dec_classifier = DecisionTreeClassifier()

X = df_encoded.drop(['Approved_Flag'], axis=1)
y = df_encoded['Approved_Flag']

label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
dec_classifier.fit(X_train, y_train)
y_pred = dec_classifier.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
print()
print(f"Accuracy: {acc_score}")
print()
precision, recall, fl_score, _=precision_recall_fscore_support(y_test, y_pred) 


for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}: ")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {fl_score[i]}")
    print()

final_acc.append(f"decision tree accuracy score is: {acc_score}")

params_grid = {
        'eta': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1,3,5],
        'subsample': [0.5, 0.7, 1],
        'colsample_bytree': [0.5, 0.7, 1],
        'gemma': [0, 1, 5],
        'lambda': [0, 1, 10],
        'alpha': [0, 1, 10]
    }

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4, use_label_encoder=False, eval_metric='mlogloss')
#perform gridsearch
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=params_grid, scoring='accuracy', cv=3, verbose=3)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score from the grid search
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
print()
print(f"Accuracy: {acc_score}")
print()
precision, recall, fl_score, _=precision_recall_fscore_support(y_test, y_pred, average=None)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}: ")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {fl_score[i]}")
    print()  
    
final_acc.append(f"XGBoost tuned accuracy score: {acc_score}")
for score in final_acc:
    print(score)













