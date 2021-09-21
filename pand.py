import pandas
data=pandas.read_csv('titanic.csv',index_col='PassengerId')
male=0
female=0
survived=0
all=0
first_class=0
for i in data['Sex']:
    if i=='male':
        male+=1
    elif i=='female':
        female+=1
print("males:",male," females:",female)
for i in data['Survived']:
    if i==1:
        survived+=1
        all+=1
    elif i==0:
        all+=1
print(survived,'   ',all)
print(survived/all)
for i in data['Pclass']:
    if i==1:
        first_class+=1
print('first/all',first_class/all)
avg=data['Age'].mean(axis=0)
median=data['Age'].median(axis=0)
print('avg=',avg)
print('median',median)
#corr=data['SibSp','Parch'].value_counts()
#'SibSp''Parch'
dataframe_1=data[['SibSp','Parch']]
corr=dataframe_1.corr()
print(corr)
name_mrs=pandas.DataFrame()
name_miss=[]
for i in data['Name']:
    if "Mrs." in i:
        name_mrs.append(i)
    elif "Miss" in i:
        name_miss.append(i)
indec=[]
print(name_mrs)
for i in name_mrs:
    counter = 0
    for j in i:

        if j=='(':
            indec.append(counter)
        counter+=1
print(indec)
name=data['Name'].value_counts().idxmax()
print('nAME=',name)
#print(name_miss)
#for i in name_mrs:
  #  j=i
