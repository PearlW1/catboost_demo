from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

# cat_features = [0,1]
# #
# train_data = [["IT互联网", "你", 1, 4, 5, 6],
#                ["IT互联网", "你", 4, 5, 6, 7],
#                ["他", "它", 30, 40, 50, 60]]
# data = pd.DataFrame(train_data)
# print(data)
# print(data.dtypes)
#  #
# train_labels = [1,1,0]


# model = CatBoostClassifier(iterations=20)
# #
# model.fit(train_data, train_labels, cat_features)
# predictions = model.predict(train_data)
# print(predictions)


#
dx = []
dy = []
cat_feat = [3, 4, 5, 8, 9, 10]
with open('./data/test', 'r') as rf:
    for line in rf:
        line_split = line.strip().split('\t')
        dy.append(int(line_split[0]))
        tmp = []
        for i in range(1, len(line_split)):
            if i not in cat_feat:
                tmp.append(int(line_split[i]))
            else:
                tmp.append(line_split[i])
        dx.append(tmp)
print(dx)


cat_features = [2, 3, 4,7, 8, 9]
dxdf = pd.DataFrame(dx)
dydf = pd.DataFrame(dy)
# dxdf = dxdf[["content_id", "author_company_id",  "author_current_position",  "author_new_top_major",  "author_new_top_profession", "author_uid", "uid", "new_top_profession", "new_top_major", "current_position", "company_id"]]
print(dxdf.dtypes)

model = CatBoostClassifier(iterations=20)
#
model.fit(dxdf, dydf, cat_features)
predictions = model.predict(dxdf)
print(predictions)
# model.get_feature_importance()

import matplotlib.pyplot as plt
import numpy as np
fea_ = model.feature_importances_
print(fea_)
print(type(fea_))
fea_name = model.feature_names_
print(fea_name)
idx_name = {}
with open('./data/feature_name.txt', 'r') as rf:
    for line in rf:
        ln = line.strip().split(' ')
        idx_name[ln[0]] = ln[1]
print(idx_name)

# fea_arr = np.array(fea_)
idx = np.argsort(fea_)
print('---idx: ', idx)
print(idx.shape)
final_feat = []
final_name = []

for i in idx:
    print('--{0}   {1}'.format(i, fea_[i]))
    final_feat.append(fea_[i])
    # print(i)
    # print(type(i))
    # print(idx_name[i])
    final_name.append(idx_name[str(i)])
print(final_feat)
print(final_name)

fig, ax = plt.subplots(figsize=(11, 11))

# Horizontal Bar Plot
ax.barh(final_name, final_feat, height=0.5)

# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)

# Add x, y gridlines
ax.grid(b=True, color='grey',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)
# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.2,
             str(round((i.get_width()), 2)),
             fontsize=9, fontweight='bold',
             color='black')
ax.set_title('feature importance' )
plt.show()

# shap_values = get_feature_importance(model)
# expected_value = shap_values[0,-1]
# shap_values = shap_values[:,:-1]
# shap.initjs()
# shap.force_plot(expected_value, shap_values[3,:], X_test.iloc[3,:])
# data = pd.read_table('./data/test', sep='\t',  header=None)
#
# # # print(data)
# print(data.head())
# X, y = data.iloc[:,1:],data.iloc[:,0]
# print(X.head())
# # #
# print(X.dtypes)
# data
# for row in X.iterrows():


# # 指定category类型的列，可以是索引，也可以是列名
# cat_features = [3, 4, 5, 8, 9, 10]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
#
#
# def auc(m, train, test):
#     return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
#                             metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))
#
# params = {'depth': [4, 7, 10],
#           'learning_rate' : [0.03, 0.1, 0.15],
#          'l2_leaf_reg': [1,4,9],
#          'iterations': [300]}
# # cb = cb.CatBoostClassifier()
# # cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv = 3)
# # cb_model.fit(train, y_train)
#
# # With Categorical features
# clf = CatBoostClassifier(eval_metric="AUC", depth=6, iterations=50, l2_leaf_reg=9, learning_rate= 0.15)
# clf.fit(X_train,y_train)
# print(auc(clf, X_train, X_test))
# #