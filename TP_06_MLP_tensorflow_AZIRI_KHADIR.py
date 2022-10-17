#!/usr/bin/env python
# coding: utf-8

# ![dth.png](attachment:dth.png)

# #### KHADIR Nadir
# #### AZIRI Abderrahmane

# # TP 6 : MLP Titanic
# 
# #### Link Github : 
# 
# ## Introduction :
# 
# 
# ##### LE MLP :
# 
# Un perceptron multicouche est un réseau neuronal reliant plusieurs couches dans un graphe dirigé, ce qui signifie que le chemin de la donnée à travers les nœuds ne va que dans un sens. Chaque nœud, à l'exception des nœuds d'entrée, possède une fonction d'activation non linéaire. Il génère un ensemble de sortie à partir d'un ensemble d'entrée.  Il est par ailleurs, considéré comme une technique d'apprentissage profond.
# 
# Le MLP est largement utilisé pour résoudre des problèmes qui nécessitent un apprentissage supervisé, ainsi que pour la recherche en neurosciences computationnelles et en traitement distribué parallèle. Les applications comprennent la reconnaissance vocale, la reconnaissance d'images et la traduction automatique. 
# 
# ##### TITANIC :
# 

# Il s’agit d’un sous-ensemble des passagers du fameux navire Titanic. L’objectif de cet exemple est de construire un modèle qui sait prédire pour un passager particulier s’il a survécu ou pas à ce drame. Ce jeu de données indique pour chaque passager les informations suivantes : 
# 
# * PassengerId : identifiant d’un passager.
# 
# * Survived : variable binaire indiquant si le passager a survécu au drame ou non.
# 
# * Pclass : variable indiquant la classe de la cabine. Elle prend ses valeurs parmi les valeurs 1, 2 et 3 qui correspondent respectivement à la première, seconde et troisième classe.
# 
# * Sex : variable indiquant le sexe du passager, Male ou Female.
# 
# * Age : variable indiquant l’âge du passager.
# 
# * SibSp : variable indiquant si le passager a des frères, des sœurs, un époux ou une épouse à bord du bateau.
# 
# * Parch : variable indiquant si le passager a des parents ou des enfants à bord du bateau.
# 
# * Fare : prix du ticket.
# 
# * Embarked : port d’embarcation (C = Cherbourg, Q = Queenstown, S = Southampton).
# 
# 
# 

# ###### Preprocessing :
# Avant d'appliquer notre model sur le jeu de données TITANIC, il va nous falloir analyser notre dataset. 

# In[257]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[461]:


path = r'/home/etudiant/Téléchargements/train.csv'
train = pd.read_csv(path)
path2 = r'/home/etudiant/Téléchargements/test.csv'
test = pd.read_csv(path2)
train.head()


# In[462]:


train.describe()


# In[464]:


train['Cabin'].describe()


# * La colonne PassengerId correspond juste un ordre arbitraire pour numériser les passagers. il est donc inutile de l'inclure dans les données de classification. la même chose est aussi valable pour les noms et prénoms des passagers qui n'influent pas sur leur chance de survie.
# 
# * la colonne cabin peut influencer la classification, mais le problème c'est une colonne qui contient beaucoup de cases vides ce qui pourra fausser la classification.
# On décide donc de la supprimer.

# In[259]:


train_db = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]


# In[260]:


train_db.dtypes


# On remplace les données catégoriques dans sex et Embarked on attribuent des valeurs numériques.

# In[261]:


train_db['Sex'] = train_db['Sex'].replace('female',0).replace('male',1)
train_db['Embarked']= train_db['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)



# On supprime les lignes avec des valeurs manqueantes. 

# In[280]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

scaler = StandardScaler()


df=train_db.dropna()
df.describe()

y=df['Survived']
X=df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

scaler.fit(X)
X = scaler.transform(X)


#scaler = MinMaxScaler()
#scaler.fit(X)
#X = scaler.transform(X)


# ##### Analyse des données et standardisation : ACP
# 
# Dans cette partie on analyse nos données pour mieux les appréhender, on applique donc une PCA pour mieux identifier les features qui ont des corrélations entre elles, et plus spécialement avec nos labels.
# 
# Par ailleurs, après multiples essais nous avons opté pour le StandardScaler() car ça donnait de meilleurs résultats à terme.

# In[290]:


from sklearn.decomposition import PCA
from sklearn import decomposition
import seaborn as sb
import numpy as np
import plotly.express as px
from numpy.linalg import eigh

pca_data = PCA()
pca_data.fit(X)
result = pca_data.transform(X)


variance = pca_data.explained_variance_ratio_ 
var=np.cumsum(np.round(pca_data.explained_variance_ratio_, decimals=3)*100)

plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,100.5)


plt.plot(var)

cov_matrix = np.cov(X, rowvar=False)



# Determine valeurs et vecteurs propres 
#
egnvalues, egnvectors = eigh(cov_matrix)


# Determine explained variance

total_egnvalues = sum(egnvalues)
var_exp = [(i/total_egnvalues) for i in sorted(egnvalues, reverse=True)]




# Plot la  variance explained et la somme cumulative de explained variance
#
import matplotlib.pyplot as plt
plt.figure()

plt.bar(range(0,len(var_exp)), var_exp, alpha=0.5, align='center', label='Explained variance individuel')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# On propose une autre approche qui consiste à illustrer une heatmap pour mieux visualiser les corrélations entre les features, ce qui nous intéresse dans ce cas et les intersections entre les features et la colonne survive.
# Dans cette carte, ce qui nous intéresse ce sont les valeurs positives ainsi que négatives.

# In[291]:


import seaborn as sns
sns.heatmap(train.corr())


# In[293]:


import seaborn as sns
sns.heatmap(df.corr())


# Survived corrélé avec fare et parch et pclass (négatif) (on n'est pas intéressé par ce qui est 0).
# 
# Survived est aussi corrélé avec sex et pclass qui est lui-même corrélé avec fare (plus grand prix).

# In[282]:


from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.datasets import mnist
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# On appliquant notre modèle MLP directement sur notre dataset nettoyé on obtient un score de 80.5%.

# In[268]:


mlp=MLPClassifier()
mlp.fit(X_train,y_train)
y_pred=mlp.predict(X_test)
#mlp.score(X_test,y_test)
print(accuracy_score(y_test,y_pred))


# ##### GRIDSEARCH  cv:
# Afin d'améliorer notre résultat on applique un gridsearchcv pour trouver les meilleures paramètres d'entrée de notre  modèle.

# In[284]:


warnings.filterwarnings("ignore")

parameter = {
'hidden_layer_sizes': [(50,50,50), (50,100,50),(100,150,150)],
'activation': ['tanh', 'relu'],
'solver': ['sgd', 'adam'],
'alpha': [0.0001,0.1,1,0.01,0.001, 0.05],
'learning_rate_init': [0.5,0.1, 0.01, 0.001],
}


clf = MLPClassifier()
qlf = GridSearchCV(clf, parameter, n_jobs=-1, verbose=11)
qlf.fit(X_train, y_train)

print(qlf.best_params_)


# In[285]:


y_pred = qlf.predict(X_test)
accuracy_score(y_test,y_pred)


# On obtient un score de 83%, soit une amélioration net de 3% par rapport à notre précèdente.  

# ##### Base de donnée biaisée :

# Après l'analyse de notre database on avait remarqué que celle-ci était très déséquilibrée et bisaisée, en d'autres termes le modèle donne plus d'importance à la classification de la classe avec plus d'éléments.
# 
# Ce biais peut influencer de nombreux algorithmes d'apprentissage automatique, conduisant certains à ignorer complètement la classe minoritaire. C'est un problème car c'est généralement sur la classe minoritaire que les prédictions sont les plus importantes.
# 
# Une approche pour résoudre le problème du déséquilibre des classes consiste à rééchantillonner de manière aléatoire l'ensemble de données d'apprentissage. Les deux principales approches de rééchantillonnage aléatoire d'un ensemble de données déséquilibré consistent à supprimer des exemples de la classe majoritaire, ce que l'on appelle le sous-échantillonnage, ou à dupliquer des exemples de la classe minoritaire, ce que l'on appelle le sur-échantillonnage.
# 
# 

# * Oversampling :

# In[308]:


df['Survived'].hist()


# Avant l'oversampling.

# In[318]:


class_2,class_1 = df.Survived.value_counts()
c2 = df[df['Survived'] == 0]
c1 = df[df['Survived'] == 1]
df_1 = c1.sample(max(class_1,class_2),replace=True)
df_2 = c2.sample(max(class_1,class_2),replace=True)
oversampled_df = pd.concat([df_1,df_2], axis=0)


# In[319]:


oversampled_df['Survived'].hist()


# Aprés l'oversampling 

# In[320]:


scaler = StandardScaler()

y=oversampled_df['Survived']
X=oversampled_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

warnings.filterwarnings("ignore")

parameter = {
'hidden_layer_sizes': [(50,50,50), (50,100,50),(100,150,150)],
'activation': ['tanh', 'relu'],
'solver': ['sgd', 'adam'],
'alpha': [0.0001,0.1,1,0.01,0.001, 0.05],
'learning_rate_init': [0.5,0.1, 0.01, 0.001],
}


clf = MLPClassifier()
qlf = GridSearchCV(clf, parameter, n_jobs=-1, verbose=11)
qlf.fit(X_train, y_train)

print(qlf.best_params_)

y_pred = qlf.predict(X_test)
print(accuracy_score(y_test,y_pred))


# ### Le score obtenu grace à cette approche est de 84.11%.

# # undersample 

# In[316]:


class_2,class_1 = df.Survived.value_counts()
c2 = df[df['Survived'] == 0]
c1 = df[df['Survived'] == 1]
df_1 = c1.sample(min(class_1,class_2))
df_2 = c2.sample(min(class_1,class_2))

undersampled_df = pd.concat([df_1,df_2], axis=0)

undersampled_df['Survived'].hist()


# In[317]:


scaler = StandardScaler()

y=undersampled_df['Survived']
X=undersampled_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

warnings.filterwarnings("ignore")

parameter = {
'hidden_layer_sizes': [(50,50,50), (50,100,50),(100,150,150)],
'activation': ['tanh', 'relu'],
'solver': ['sgd', 'adam'],
'alpha': [0.0001,0.1,1,0.01,0.001, 0.05],
'learning_rate_init': [0.5,0.1, 0.01, 0.001],
}


clf = MLPClassifier()
qlf = GridSearchCV(clf, parameter, n_jobs=-1, verbose=11)
qlf.fit(X_train, y_train)

print(qlf.best_params_)

y_pred = qlf.predict(X_test)
print('under sample score',accuracy_score(y_test,y_pred))


# ### Le score obtenu grace à cette approche est de 81.89%.

# ### On ajoute la colonne ticket + Oversampling: 

# In[448]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
all_db = train

all_db['Sex'] = all_db['Sex'].replace('female',0).replace('male',1)
all_db['Embarked']= all_db['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)

x = all_db['Age']
all_db['Age'] = label_encoder.fit_transform(x)

x = all_db['Ticket']
all_db['Ticket'] = label_encoder.fit_transform(x)

all_db=all_db.dropna()


all_db = all_db[['Pclass', 'Sex', 'Age' , 'SibSp' , 'Parch' ,'Ticket','Fare','Embarked', 'Survived']]


class_2,class_1 = all_db.Survived.value_counts()
c2 = all_db[all_db['Survived'] == 0]
c1 = all_db[all_db['Survived'] == 1]
df_1 = c1.sample(max(class_1,class_2),replace=True)
df_2 = c2.sample(max(class_1,class_2),replace=True)
oversampled_df = pd.concat([df_1,df_2], axis=0)



scaler = StandardScaler()

y=oversampled_df['Survived']
X=oversampled_df[['Pclass', 'Sex', 'Age' , 'SibSp' , 'Parch' ,'Ticket','Fare','Embarked']]


scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

warnings.filterwarnings("ignore")

parameter = {
'hidden_layer_sizes': [(50,50,50), (50,100,50),(100,150,150)],
'activation': ['tanh', 'relu'],
'solver': ['sgd', 'adam'],
'alpha': [0.0001,0.1,1,0.01,0.001, 0.05],
'learning_rate_init': [0.5,0.1, 0.01, 0.001],
}


clf = MLPClassifier()
qlf = GridSearchCV(clf, parameter, n_jobs=-1, verbose=11)
qlf.fit(X_train, y_train)

print(qlf.best_params_)

y_pred = qlf.predict(X_test)
print(accuracy_score(y_test,y_pred))




# ## Le score obtenu grace à cette approche est de 87.03%.

# # Avec tensorflow : 

# In[704]:


label_encoder = LabelEncoder()
all_db = train

all_db['Sex'] = all_db['Sex'].replace('female',0).replace('male',1)
all_db['Embarked']= all_db['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)

x = all_db['Age']
all_db['Age'] = label_encoder.fit_transform(x)

x = all_db['Ticket']
all_db['Ticket'] = label_encoder.fit_transform(x)

all_db=all_db.dropna()


all_db = all_db[['Pclass', 'Sex', 'Age' , 'SibSp' , 'Parch' ,'Ticket','Fare','Embarked', 'Survived']]


class_2,class_1 = all_db.Survived.value_counts()
c2 = all_db[all_db['Survived'] == 0]
c1 = all_db[all_db['Survived'] == 1]
df_1 = c1.sample(max(class_1,class_2),replace=True)
df_2 = c2.sample(max(class_1,class_2),replace=True)
oversampled_df = pd.concat([df_1,df_2], axis=0)



scaler = StandardScaler()

y=oversampled_df['Survived']
X=oversampled_df[['Pclass', 'Sex', 'Age' , 'SibSp' , 'Parch' ,'Ticket','Fare','Embarked']]


scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[705]:


tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
  
model.add(tf.keras.Input(shape=(X_train.shape[0],X_train.shape[1])))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.1)) # eviter l'overfitting 
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


metric = tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
)


model.compile(loss=tf.keras.losses.CategoricalHinge(), optimizer='adam', metrics=metric)

model.fit(X_train,y_train, epochs=500,validation_split=0.05)


# In[706]:


print('Score : ', model.evaluate(X_test,  y_test)[1])


# On a utilisé 4 couches, la première est l'input layer 
# La deuxième et la troisième sont les couches cachées 'hidden layers', le choix du nombre de neurones par couche est arbitraire.
# Et la troisième est l'output layer, sachant qu'on a une décision binaire pour notre output, on a choisi une fonction sigmoïde afin de nous rendre notre résultat [0,1].

# In[707]:


from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

def classifier():
    tf.keras.backend.clear_session()
    model=tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(X_train.shape[0],X_train.shape[1])))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.2)) # eviter l'overfitting 
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    metric = tf.keras.metrics.BinaryAccuracy(
        name="binary_accuracy", dtype=None, threshold=0.5 )
    model.compile(loss=tf.keras.losses.CategoricalHinge(), optimizer='Nadam', metrics=['accuracy'])
    return model


model=KerasClassifier(build_fn=classifier)




# define the grid search parameters

param_grid = {
    
    'batch_size' :  [80, 100],
    'epochs' : [10, 50, 100, 200,300],
    'shuffle':[True,False],
    'validation_split':[0.05, 0.1,0.2],
    'use_multiprocessing':[True], 
    
}




gs=GridSearchCV(estimator=model,param_grid=param_grid, cv=3)
# now fit the dataset to the GridSearchCV object. 
gs = gs.fit(X_train, y_train)


# In[711]:


best_params=gs.best_params_
y_pred = gs.predict(X_test)

print('Score : ', accuracy_score(y_test,y_pred))
print('best_params  : '  ,best_params)


# In[712]:


print('Score : ', accuracy_score(y_test,y_pred))


# best_params  :  {'batch_size': 100, 'epochs': 200, 'shuffle': False, 'use_multiprocessing': True, 'validation_split': 0.1}

# # le meilleur score obtenue pour notre modèle est de 94.02%

# ## On essai de trouver le nombre de neuronnes par couche ideal : 

# In[499]:


layers = [32, 64, 128, 256, 512, 1024]

score = []

for i in layers: 
    for j in layers: 
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(X_train.shape[0],X_train.shape[1])))
        model.add(tf.keras.layers.Dense(i, activation='relu'))
        model.add(tf.keras.layers.Dense(j, activation='relu'))
        #model.add(tf.keras.layers.Dropout(0.2)) # eviter l'overfitting 
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


        metric = tf.keras.metrics.BinaryAccuracy(
            name="binary_accuracy", dtype=None, threshold=0.5
        )


        model.compile(loss=tf.keras.losses.CategoricalHinge(), optimizer='adam', metrics=metric)

        history = model.fit(X_train,y_train, epochs=200,validation_split=0.1)
        
        print('Score : ', model.evaluate(X_test,  y_test)[1])
        score.append([i , j , model.evaluate(X_test,  y_test)[1]])
        
        
        


# In[544]:


score = pd.DataFrame(score)
ALL_LAYER_NUMBER = (score.where(score[2] == max(score[2])).dropna())
print('Nombre de neuronnes dans les deux couches optimales ', 
      ALL_LAYER_NUMBER.where(ALL_LAYER_NUMBER[0]+ALL_LAYER_NUMBER[1] == 
                             min(ALL_LAYER_NUMBER[0]+ALL_LAYER_NUMBER[1])).dropna()[[0,1,2]])


# In[530]:


ALL_LAYER_NUMBER[0]+ALL_LAYER_NUMBER[1]


# In[726]:


tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
  
model.add(tf.keras.Input(shape=(X_train.shape[0],X_train.shape[1])))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.1)) # eviter l'overfitting 
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


metric = tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
)


model.compile(loss=tf.keras.losses.CategoricalHinge(), optimizer='adam', metrics=metric)

model.fit(X_train,y_train,batch_size = 100, epochs=200,shuffle = False, use_multiprocessing = True, 
          validation_split=0.1)


# In[727]:


print('Score : ', model.evaluate(X_test,  y_test)[1])


# ### Le nombre de neuronnes par couche n'a pas l'air de modifier le score final

# In[737]:


tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
  
model.add(tf.keras.Input(shape=(X_train.shape[0],X_train.shape[1])))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.1)) # eviter l'overfitting 
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


metric = tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
)


model.compile(loss=tf.keras.losses.CategoricalHinge(), optimizer='Nadam', metrics=metric)

model.fit(X_train,y_train,batch_size = 100, epochs=200,shuffle = False, use_multiprocessing = True, 
          validation_split=0.1)


# In[738]:


print('Score : ', model.evaluate(X_test,  y_test)[1])


# In[748]:


tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
  
model.add(tf.keras.Input(shape=(X_train.shape[0],X_train.shape[1])))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.1)) # eviter l'overfitting 
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


metric = tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
)


model.compile(loss=tf.keras.losses.CategoricalHinge(), optimizer='Nadam', metrics=metric)

model.fit(X_train,y_train,batch_size = 100, epochs=200,shuffle = False, use_multiprocessing = True, 
          validation_split=0.1)


# In[749]:


print('Score : ', model.evaluate(X_test,  y_test)[1])


# In[ ]:




