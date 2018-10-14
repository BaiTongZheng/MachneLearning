#coding:utf-8
import pandas as pd 
import numpy as np 
from pandas import Series
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

data_train = pd.read_csv("/home/baitong/Data/all/train.csv")
# print(data_train)
# print(data_train.info())

# plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
# data_train.Survived.value_counts().plot(kind='bar')# 柱状图 
# plt.title("rescue (1 was Survived)") # 标题
# plt.ylabel(u"Num")  

# plt.subplot2grid((2,3),(0,1))
# data_train.Pclass.value_counts().plot(kind="bar")
# plt.ylabel(u"Num")
# plt.title(u"level passenger")

# plt.subplot2grid((2,3),(0,2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel(u"Age")                         # 设定纵坐标名称
# plt.grid(b=True, which='major', axis='y') 
# plt.title(u"Age  (1 was Survived)")


# plt.subplot2grid((2,3),(1,0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"Age")# plots an axis lable
# plt.ylabel(u"density") 
# plt.title(u"by level age ")
# plt.legend((u'level 1', u'level 2',u' level 3'),loc='best') # sets our legend for our graph.


# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u"Embarked")
# plt.ylabel(u"Num")  
# plt.show()

#看看各乘客按客舱等级的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数

# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({u'Survived':Survived_1, u'unSurvived':Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u"sur distri on diff level")
# plt.xlabel(u"passenger level") 
# plt.ylabel(u"Num") 
# plt.show()
#查看乘客按性别的情况
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df=pd.DataFrame({u'male':Survived_m, u'female':Survived_f})
# df.plot(kind='bar', stacked=True)
# plt.title(u"sur distri on gender")
# plt.xlabel(u"gender") 
# plt.ylabel(u"num")
# plt.show()

#查看乘客按性别和客舱等级的状况
# plt.title(u"sur distri by gender and level")

# ax1=fig.add_subplot(141)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female high class", color='#FA2479')
# # ax1.set_xticklabels([u"Survived", u"unSurvived"], rotation=0)
# ax1.legend([u"female/level1,2"], loc='best')

# ax2=fig.add_subplot(142, sharey=ax1)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
# # ax2.set_xticklabels([u"unSurvived", u"Survived"], rotation=0)
# plt.legend([u"female/level3"], loc='best')

# ax3=fig.add_subplot(143, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
# # ax3.set_xticklabels([u"unSurvived", u"Survived"], rotation=0)
# plt.legend([u"male/level1,2"], loc='best')

# ax4=fig.add_subplot(144, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male, low class', color='steelblue')
# # ax4.set_xticklabels([u"unSurvived", u"Survived"], rotation=0)
# plt.legend([u"male/level3"], loc='best')

# plt.show()
#获救情况与港口的关系
# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({u'Survived':Survived_1, u'unSurvived':Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u"by hourbor")
# plt.xlabel(u"Embarked") 
# plt.ylabel(u"Num") 

# plt.show()
#堂兄妹数量/父母孩子数量 对生存的影响
# g = data_train.groupby(['SibSp','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)

# g = data_train.groupby(['Parch','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)

#ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴把
#cabin只有204个乘客有值，我们先看看它的一个分布
# print(data_train.Cabin.value_counts())

#有无cabin对存活率的影响（cabin数据并不是完全可靠，可能存在许多丢失值）
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数

# Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df=pd.DataFrame({u'have':Survived_cabin, u'not have':Survived_nocabin}).transpose()
# df.plot(kind='bar', stacked=True)
# plt.title(u"by cabin")
# plt.xlabel(u"dose have Cabin") 
# plt.ylabel(u"Num")
# plt.show()
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # y即目标年龄
    y = known_age[:, 0]
    # print(y)
    # X即特征属性值
    X = known_age[:, 1:]
    # print(X)
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    # print(predictedAges)
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)

data_train = set_Cabin_type(data_train)
# print('*'*20)
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# print(df[:5])
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df[['Age']])
# print("age_scale_param")
# print(age_scale_param)
df['Age_scaled'] = scaler.fit_transform(df[['Age']], age_scale_param)
# print("df['Age_scaled']")
# print(df['Age_scaled'][:5])
fare_scale_param = scaler.fit(df[['Fare']])
# print("fare_scale_param")
# print(fare_scale_param)
df['Fare_scaled'] = scaler.fit_transform(df[['Fare']], fare_scale_param)
# print("df['Fare_scaled']")
# print(df['Fare_scaled'][:5])
# 用正则取出我们要的属性值
# train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# train_np = train_df.values
# # y即Survival结果
# y = train_np[:, 0]
# # X即特征属性值
# X = train_np[:, 1:]
# fit到RandomForestRegressor之中
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# clf.fit(X, y)

###############################################################
#处理test数据
############################################################
# data_test = pd.read_csv("/home/baitong/Data/all/test.csv")
# data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# # 接着我们对test_data做和train_data中一致的特征变换
# # 首先用同样的RandomForestRegressor模型填上丢失的年龄
# tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
# null_age = tmp_df[data_test.Age.isnull()].values
# # 根据特征属性X预测年龄并补上
# X = null_age[:, 1:]
# predictedAges = rfr.predict(X)
# data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

# data_test = set_Cabin_type(data_test)
# dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
# dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
# dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
# dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


# df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
# df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age']], age_scale_param)
# df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Fare']], fare_scale_param)
####################test##########################
# test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(test)
# result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
# result.to_csv("/home/baitong/pywork/Titannic_Machine_Learnning/logistic_regression_predictions.csv", index=False)

######查看第一次训练后的参数####
# print(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}))
 #简单看看打分情况
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# X = all_data.values[:,1:]
# y = all_data.values[:,0]
# print(cross_val_score(clf, X, y, cv=5))
# 分割数据，按照 训练数据:cv数据 = 7:3的比例
split_train, split_cv = train_test_split(df, test_size=0.3, random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# 生成模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_df.values[:,1:], train_df.values[:,0])

# 对cross validation数据进行预测

cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(cv_df.values[:,1:])
origin_data_train = pd.read_csv("/home/baitong/Data/all/train.csv")
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:,0]]['PassengerId'].values)]
bad_cases.to_csv("/home/baitong/pywork/Titannic_Machine_Learnning/bad_cases.csv", index=False)
print(pd.DataFrame({"columns":list(cv_df.columns)[1:], "coef":list(clf.coef_.T)}))