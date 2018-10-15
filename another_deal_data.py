
import pandas as pd 
import numpy as np 
from pandas import DataFrame
from patsy import dmatrices
from patsy import dmatrix
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
import string

filename = "/home/baitong/Data/all"
data_train = pd.read_csv(filename+"/train.csv")

le = preprocessing.LabelEncoder()

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    #print(big_string)
    return np.nan

def clean_and_munge_data(df):
    #处理缺省值
    df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
    #处理一下名字，生成Title字段
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))

    #处理特殊的称呼，全处理成mr, mrs, miss, master
    def replace_titles(x):
        title=x['Title']
        if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Master']:
            return 'Master'
        elif title in ['Countess', 'Mme','Mrs']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms','Miss']:
            return 'Miss'
        elif title =='Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        elif title =='':
            if x['Sex']=='Male':
                return 'Master'
            else:
                return 'Miss'
        else:
            return title

    df['Title']=df.apply(replace_titles, axis=1)
    #print(df['Title'])
    #看看家族是否够大，咳咳
    df['Family_Size']=df['SibSp']+df['Parch']
    df['Family']=df['SibSp']*df['Parch']

    df.loc[ (df.Fare.isnull())&(df.Pclass==1),'Fare'] =np.median(df[df['Pclass'] == 1]['Fare'].dropna())
    df.loc[ (df.Fare.isnull())&(df.Pclass==2),'Fare'] =np.median( df[df['Pclass'] == 2]['Fare'].dropna())
    df.loc[ (df.Fare.isnull())&(df.Pclass==3),'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())

    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    df['AgeFill']=df['Age']
    mean_ages = np.zeros(4)
    mean_ages[0]=np.average(df[df['Title'] == 'Miss']['Age'].dropna())
    mean_ages[1]=np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
    mean_ages[2]=np.average(df[df['Title'] == 'Mr']['Age'].dropna())
    mean_ages[3]=np.average(df[df['Title'] == 'Master']['Age'].dropna())
    df.loc[ (df.Age.isnull()) & (df.Title == 'Miss') ,'AgeFill'] = mean_ages[0]
    df.loc[ (df.Age.isnull()) & (df.Title == 'Mrs') ,'AgeFill'] = mean_ages[1]
    df.loc[ (df.Age.isnull()) & (df.Title == 'Mr') ,'AgeFill'] = mean_ages[2]
    df.loc[ (df.Age.isnull()) & (df.Title == 'Master') ,'AgeFill'] = mean_ages[3]

    df['AgeCat']=df['AgeFill']
    df.loc[ (df.AgeFill<=10) ,'AgeCat'] = 'child'
    df.loc[ (df.AgeFill>60),'AgeCat'] = 'aged'
    df.loc[ (df.AgeFill>10) & (df.AgeFill <=30) ,'AgeCat'] = 'adult'
    df.loc[ (df.AgeFill>30) & (df.AgeFill <=60) ,'AgeCat'] = 'senior'

    df.Embarked = df.Embarked.fillna('S')


    df.loc[ df.Cabin.isnull()==True,'Cabin'] = 0.5
    df.loc[ df.Cabin.isnull()==False,'Cabin'] = 1.5

    df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)

    #Age times class

    df['AgeClass']=df['AgeFill']*df['Pclass']
    df['ClassFare']=df['Pclass']*df['Fare_Per_Person']


    df['HighLow']=df['Pclass']
    df.loc[ (df.Fare_Per_Person<8) ,'HighLow'] = 'Low'
    df.loc[ (df.Fare_Per_Person>=8) ,'HighLow'] = 'High'

    le.fit(df['Sex'] )
    x_sex=le.transform(df['Sex'])
    df['Sex']=x_sex.astype(np.float)

    le.fit( df['Ticket'])
    x_Ticket=le.transform( df['Ticket'])
    df['Ticket']=x_Ticket.astype(np.float)

    le.fit(df['Title'])
    x_title=le.transform(df['Title'])
    df['Title'] =x_title.astype(np.float)

    le.fit(df['HighLow'])
    x_hl=le.transform(df['HighLow'])
    df['HighLow']=x_hl.astype(np.float)


    le.fit(df['AgeCat'])
    x_age=le.transform(df['AgeCat'])
    df['AgeCat'] =x_age.astype(np.float)

    le.fit(df['Embarked'])
    x_emb=le.transform(df['Embarked'])
    df['Embarked']=x_emb.astype(np.float)



    df = df.drop(['PassengerId','Name','Age','Cabin'], axis=1) #remove Name,Age and PassengerId


    return df
df=clean_and_munge_data(data_train)

formula_ml='Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size' 
# df.to_csv(filename+"/clean_and_munge_data.csv")
y_train, x_train = dmatrices(formula_ml, data=df, return_type='dataframe')

x_train.to_csv(filename+"/x_train.csv")
y_train.to_csv(filename+"/y_train.csv")

train_label = np.asarray(y_train).ravel()
# print(y_train)
scaler = preprocessing.StandardScaler()
fare_scale_param1 = scaler.fit(x_train[['Fare_Per_Person']])
x_train['sFare_per_scaled'] = scaler.fit_transform(x_train[['Fare_Per_Person']], fare_scale_param1)
fare_scale_param2 = scaler.fit(x_train[['Fare']])
x_train['sFare_scaled'] = scaler.fit_transform(x_train[['Fare']], fare_scale_param2)
family_scale_param = scaler.fit(x_train[['Family_Size']])
x_train['Family_Size_scaled'] = scaler.fit_transform(x_train[['Family_Size']], family_scale_param)

train_feature = x_train.filter(regex='C(Title)|C(AgeCat)*|sFare_.*|Sex|Pclass|Family_Size_*')


dummies_Sex = pd.get_dummies(train_feature['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(train_feature['Pclass'], prefix= 'Pclass')

train_feature = pd.concat([train_feature, dummies_Sex, dummies_Pclass], axis=1)
train_feature.drop(['Pclass', 'Sex','Family_Size'], axis=1, inplace=True)
train_feature.to_csv(filename+"/train_feature.csv")

########cross validation#############
X_train, X_test, Y_train, Y_test = train_test_split(train_feature, train_label, test_size=0.2,random_state=0)
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X_train, Y_train)
print("cross validation")
print(clf.score(X_test,Y_test))
#####################################


######### testdata preprocessing ###########
data_t = pd.read_csv(filename+"/test.csv")
data_test = clean_and_munge_data(data_t)
formula_t='Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size'


data_test = dmatrix(formula_t, data=data_test, return_type='dataframe')

data_test.to_csv(filename+"/test_deal.csv")
fare_scale_param1 = scaler.fit(data_test[['Fare_Per_Person']])
data_test['sFare_per_scaled'] = scaler.fit_transform(data_test[['Fare_Per_Person']], fare_scale_param1)
fare_scale_param2 = scaler.fit(data_test[['Fare']])
data_test['sFare_scaled'] = scaler.fit_transform(data_test[['Fare']], fare_scale_param2)
family_scale_param = scaler.fit(data_test[['Family_Size']])
data_test['Family_Size_scaled'] = scaler.fit_transform(data_test[['Family_Size']], family_scale_param)


test_feature = data_test.filter(regex='C(Title)|C(AgeCat)*|sFare_.*|Sex|Pclass|Family_Size_*')
dummies_Sex = pd.get_dummies(test_feature['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(test_feature['Pclass'], prefix= 'Pclass')

test_feature = pd.concat([test_feature, dummies_Sex, dummies_Pclass], axis=1)
test_feature.drop(['Pclass', 'Sex','Family_Size'], axis=1, inplace=True)
test_feature.to_csv(filename+"/test_feature.csv")




##########logistic regression##################
    # test_prediction = clf.predict(test_feature)
    # outdata = pd.DataFrame({'PassengerId':data_t['PassengerId'].values,'Survived':test_prediction.astype(np.int32)})
    # print(outdata)
    # outdata.to_csv("/home/baitong/pywork/Titannic_Machine_Learnning/submission.csv", index=False)

########## Bagging ####################
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
# bagging_clf.fit(train_feature, train_label)
# test_prediction = bagging_clf.predict(test_feature)
# outdata = pd.DataFrame({'PassengerId':data_t['PassengerId'].values,'Survived':test_prediction.astype(np.int32)})
# outdata.to_csv("/home/baitong/pywork/Titannic_Machine_Learnning/submission.csv", index=False)