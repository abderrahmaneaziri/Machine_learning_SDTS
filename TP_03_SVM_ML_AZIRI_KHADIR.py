#!/usr/bin/env python
# coding: utf-8

# ![dth.png](attachment:dth.png)

# #### KHADIR Nadir
# #### AZIRI Abderrahmane

# # TP 5 : Machines à vecteurs de support (SVM)
# 
# ## Introduction :
# 
# Une machine à vecteurs de support (SVM) est un classificateur discriminant formellement défini par un hyperplan de séparation. En d'autres termes, étant donné des données d'apprentissage étiquetées (apprentissage supervisé), l'algorithme produit un hyperplan optimal qui catégorise les nouveaux exemples. 
# Dans un espace à deux dimensions, cet hyperplan est une ligne qui divise un plan en deux parties où chaque classe se trouve de part et d'autre.
# 
# 

# En premier lieu, on importe les données de notre dataset diabètes et on les illustre grace à "Seaborn" qui nous permet d'afficher chaque deux features sur un plan à 2D.

# In[3]:


import numpy as np 
import pandas as pd
import seaborn as sb

df = pd.read_csv(r'/home/etudiant/Téléchargements/diabetes.csv')
data = df[['Pregnancies' ,'Glucose', 'BloodPressure','SkinThickness' ,'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']]
y = np.array(df['Outcome'])
X = np.array(data)

sb.pairplot(df, hue="Outcome")


# Afin de continuer dans ce TP il nous faut sélectionner seulement deux features qui auront une grande importance pour nous permettre de faire une classification convenable avec un modèle linéaire.
# Dans cette optique, il existe de nombreux types de scores d'importance des caractéristiques, bien que les exemples populaires incluent les scores de corrélation statistique, les coefficients calculés dans le cadre de modèles linéaires, les arbres de décision et les scores d'importance de permutation.
# 
# Les scores d'importance des caractéristiques jouent un rôle important dans un projet de modélisation prédictive, notamment en fournissant un aperçu des données, un aperçu du modèle et une base pour la réduction de la dimensionnalité et la sélection des caractéristiques qui peuvent améliorer l'efficacité d'un modèle prédictif sur le problème.

# * On commence par une analyse PCA ou pour être plus précis de la covariance afin de déterminer la quantité d'informations que contiendraient seulement deux features.

# In[4]:


from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eigh

# On doit toujour normaliser pour un PCA

sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

pca = PCA()
pca.fit(X_std)
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)


px.area(
    x=range(1, exp_var_cumul.shape[0] + 1),
    y=exp_var_cumul,
    labels={"x": "# Components", "y": "Explained Variance"}
)


# In[5]:



# On construit la matrice de covarience 
cov_matrix = np.cov(X_std, rowvar=False)



# Determine valeurs et vecteurs propres 
#
egnvalues, egnvectors = eigh(cov_matrix)


# Determine explained variance

total_egnvalues = sum(egnvalues)
var_exp = [(i/total_egnvalues) for i in sorted(egnvalues, reverse=True)]




# Plot la  variance explained et la somme cumulative de explained variance
#
import matplotlib.pyplot as plt

plt.bar(range(0,len(var_exp)), var_exp, alpha=0.5, align='center', label='Explained variance individuel')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# * Deux features nous permettent de représenter ~47% de nos data d'après l'analyse PCA, afin de trouver lesquelles de ces features seront choisies on doit trouver l'importance de chacune d'elles.
# 

# In[6]:


from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot


features_names = ['Pregnancies' ,'Glucose', 'BloodPressure',
                  'SkinThickness' ,'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']

model = DecisionTreeClassifier()
model.fit(X, y)
importance = model.feature_importances_


pyplot.barh([x for x in features_names], importance)
pyplot.show()


# * Pour un classifieur de type arbre de décision, la feature qui semble être la plus importante est le glucose avec derrière elle le BMI. 
# 
# * Feature_importances_ n'est pas disponible pour le SVM, par contre on peut avoir le "coef" qui est le poids donné au feature, ce qui nous permet d'avoir un aperçu de l'importance de cette dernière.
# 

# In[7]:


import matplotlib.pyplot as plt
from sklearn import svm, datasets


features_names = ['Pregnancies' ,'Glucose', 'BloodPressure',
                  'SkinThickness' ,'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']


model = svm.LinearSVC(C=1)
model.fit(X, y)

plt.figure()
plt.barh([x for x in features_names], model.coef_[0])
plt.show()


# * Pour cette méthode on remarque que les poids donnés aux features DiabetesPedigreeFunction et Pregnancies sont les plus élevés. Ceci reste juste une méthode d'approche afin d'avoir un aperçu des deux features les plus signifiantes afin de les sélectionner.
# 
# On remarque que les features avec le plus d'importance changent en fonction du classifieur. On va donc prendre deux features [ DiabetesPedigreeFunction, Pregnancies] afin de continuer, mais cela ne certifie pas que ces deux dernières nous donnent une classification suffisante. si le score est trop faible on utilisera [glucose, BMI] pour tester.

# ### On remarque aussi que le score varie beacoup si on relance a chaque fois. On essai de voir la variation en fonction de chaque train_test_split.

# In[215]:


score = []
for i in range (1,700):
   data = df[['Pregnancies' ,'Glucose', 'BloodPressure',
              'SkinThickness' ,'Insulin','BMI', 
              'DiabetesPedigreeFunction', 'Age']]

   y = np.array(df['Outcome'])
   X = np.array(data)

   X= np.array(data)
   sc = StandardScaler()
   sc.fit(X)
   X = sc.transform(X)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)
   clf = svm.SVC(C=0.1, gamma=0.1, kernel='sigmoid')
   clf.fit(X_train, y_train)
   score.append(clf.score(X_test, y_test))
plt.figure()
plt.plot([x for x in range(1,700)],score)
plt.xlabel('Itteration')
plt.ylabel('Score test')
print('Écart-type : ', np.std(np.array(score)))


# #### RQ : les iteration du train_test_split influent sur le score obtenu.

# ### Question Calculez le score d’échantillons bien classifiés sur le jeu de données de test.

# In[243]:


from sklearn.model_selection import train_test_split

data_2 = df[['Pregnancies' ,'DiabetesPedigreeFunction']]

X_2 = np.array(data_2)

X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size=0.50, random_state = 0 )


# In[244]:


C = 1.0 
lin_svc = svm.LinearSVC(C=C)
lin_svc.fit(X_train, y_train)

lin_svc.score(X_test, y_test)


# In[245]:


# Créer la surface de décision discretisée
x_min, x_max = X_2[:, 0].min() - 1, X_2[:, 0].max() + 1
y_min, y_max = X_2[:, 1].min() - 1, X_2[:, 1].max() + 1
# Pour afficher la surface de décision on va discrétiser l'espace avec un pas h
h = max((x_max - x_min) / 100, (y_max - y_min) / 100)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Surface de décision
Z = lin_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# Afficher aussi les points d'apprentissage
plt.scatter(X_train[:, 0], X_train[:, 1], label="train", edgecolors='k',
c=y_train, cmap=plt.cm.coolwarm)
plt.scatter(X_test[:, 0], X_test[:, 1], label="test", marker='*', c=y_test,
cmap=plt.cm.coolwarm)
plt.xlabel('Pregnancies')
plt.ylabel('DiabetesPedigreeFunction')
plt.title("LinearSVC")


# * On resait avec les features les plus importantes trouvées pour la décision tree classifier

# In[249]:


from sklearn.model_selection import train_test_split

data_3 = df[['Glucose' ,'BMI']]

X_3 = np.array(data_3)

X_train, X_test, y_train, y_test = train_test_split(X_3, y, test_size=0.50, random_state = 0)
C = 1.0 
lin_svc = svm.LinearSVC(C=C)
lin_svc.fit(X_train, y_train)
lin_svc.score(X_test, y_test)


# In[252]:


# Créer la surface de décision discretisée
x_min, x_max = X_2[:, 0].min() - 1, X_2[:, 0].max() + 1
y_min, y_max = X_2[:, 1].min() - 1, X_2[:, 1].max() + 1
# Pour afficher la surface de décision on va discrétiser l'espace avec un pas h
h = max((x_max - x_min) / 100, (y_max - y_min) / 100)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Surface de décision
Z = lin_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# Afficher aussi les points d'apprentissage
plt.scatter(X_train[:, 0], X_train[:, 1], label="train", edgecolors='k',
c=y_train, cmap=plt.cm.coolwarm)
plt.scatter(X_test[:, 0], X_test[:, 1], label="test", marker='*', c=y_test,
cmap=plt.cm.coolwarm)
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title("LinearSVC")


# * On calcul le score pour toutes les paires possible de features de notre dataset :

# In[253]:


features_names = ['Pregnancies' ,'Glucose', 'BloodPressure',
                  'SkinThickness' ,'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
score = []
for i in features_names: 
    for j in features_names: 
        if i!=j: 
            data = df[[i,j]]
            X_4 = np.array(data)
            sc = StandardScaler()
            sc.fit(X_4)
            X_4 = sc.transform(X_4)
            X_train, X_test, y_train, y_test = train_test_split(X_4, y, test_size=0.50, random_state = 0)

            lin_svc = svm.LinearSVC(C=1)
            lin_svc.fit(X_train, y_train)
            score.append([i,j,lin_svc.score(X_test, y_test)])


# In[261]:


#score=np.array(score[:])
sco=pd.DataFrame(score)
sc=sco[2].max()

index = sco.index
condition = sco[2] == sc
features_indices = index[condition]
features_indices_list = features_indices.tolist()
print(score[features_indices_list[0]])
print(score[features_indices_list[1]])


# On obtient les deux features 'Glucose' et 'BMI' avec un score de 0.75
# 
# On illustre leur surface de décision

# In[262]:


from sklearn.model_selection import train_test_split

data_3 = df[['Glucose' ,'BMI']]

X_3 = np.array(data_3)

sc = StandardScaler()
sc.fit(X_3)
X_3 = sc.transform(X_3)

X_train, X_test, y_train, y_test = train_test_split(X_3, y, test_size=0.5, random_state = 0)
C = 1.0 
lin_svc = svm.LinearSVC(C=C)
lin_svc.fit(X_train, y_train)
lin_svc.score(X_test, y_test)


x_min, x_max = X_3[:, 0].min() - 1, X_3[:, 0].max() + 1
y_min, y_max = X_3[:, 1].min() - 1, X_3[:, 1].max() + 1
# Pour afficher la surface de décision on va discrétiser l'espace avec un pas h
h = max((x_max - x_min) / 100, (y_max - y_min) / 100)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Surface de décision
Z = lin_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# Afficher aussi les points d'apprentissage
plt.scatter(X_train[:, 0], X_train[:, 1], label="train", edgecolors='k', c=y_train, cmap=plt.cm.coolwarm)
plt.scatter(X_test[:, 0], X_test[:, 1], label="test", marker='*', c=y_test, cmap=plt.cm.coolwarm)
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title("Score: "+str(lin_svc.score(X_test, y_test)))


# ### Question : Testez différentes valeurs pour le paramètre C. Comment la frontière de décision évolue en fonction de C ?

# In[269]:


for i in [1,5,20,200]: 
    
    data = df[['BMI','Glucose']]
    X_5 = np.array(data)
    sc = StandardScaler()
    sc.fit(X_5)
    X_5 = sc.transform(X_5)
    X_train, X_test, y_train, y_test = train_test_split(X_5, y, test_size=0.50, random_state = 0)
    C = i
    lin_svc = svm.LinearSVC(C=C)
    lin_svc.fit(X_train, y_train)
    
    
    # Créer la surface de décision discretisée
    x_min, x_max = X_5[:, 0].min() - 1, X_5[:, 0].max() + 1
    y_min, y_max = X_5[:, 1].min() - 1, X_5[:, 1].max() + 1
    # Pour afficher la surface de décision on va discrétiser l'espace avec un pas h
    h = max((x_max - x_min) / 100, (y_max - y_min) / 100)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Surface de décision
    Z = lin_svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # Afficher aussi les points d'apprentissage
    
    plt.scatter(X_train[:, 0], X_train[:, 1], label="train", edgecolors='k',
    c=y_train, cmap=plt.cm.coolwarm)
    plt.scatter(X_test[:, 0], X_test[:, 1], label="test", marker='*', c=y_test,
    cmap=plt.cm.coolwarm)
    plt.xlabel('BMI')
    plt.ylabel('Glucose')
    plt.title("LinearSVC avec C=" +str(i)+'    Et un score de : '+str(lin_svc.score(X_test, y_test)))


# Le paramètre C indique à l'optimisation SVM dans quelle mesure nous voulons éviter de mal classer chaque point.
# Pour les grandes valeurs de C, l'optimisation choisira un hyperplan à marge réduite si cet hyperplan permet de mieux classer correctement tous les points de formation. À l'inverse, une très petite valeur de C amènera l'optimiseur à rechercher un hyperplan de séparation à marge plus importante, même si cet hyperplan classe mal davantage de points. Pour de très petites valeurs de C, nous devrions obtenir des exemples mal classés, souvent même si nos données d'apprentissage sont linéairement séparables.
# 

# ### Question: D’après la visualisation ci-dessus, ce modèle vous paraît-il adapté au problème ? 
# ### Si non, que peut-on faire pour l’améliorer ?
# 
# Non, le modèle nous paraît pas adapté au problème et pour cause, on constate que les données sont très mélangés entre elles et une séparation sur deux dimensions ne parviendra pas à fournir une séparation convenable.
# 
# Dans le but d'améliorer la performance de notre modèle il est nécessaire de prendre en compte d'autres features.
# Il faudra aussi changer le type du kernel employé, une méthode non linéaire donnera probablement de meilleurs résultats de séparation.

# ### Bonus : Application de models non-linéaires : 
# 
# Afin d'améliorer notre score on décide d'appliquer un GridSearch sur notre classifieur, on s'attend à obtenir comme meilleur paramètre un kernel non lineaire.

# In[271]:


data_3 = df[['Glucose' ,'BMI']]

X_3 = np.array(data_3)

sc = StandardScaler()
sc.fit(X_3)
X_3 = sc.transform(X_3)

X_train, X_test, y_train, y_test = train_test_split(X_3, y, test_size=0.5, random_state = 0)


param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']} 
  
grid = GridSearchCV(svm.SVC(), param_grid)
  
# fitting the model for grid search
grid.fit(X_train, y_train)


# print best parameter after tuning
print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

print("Score: "+str(grid.score(X_test, y_test)))


# In[272]:


grid.best_score_


# ### Question Réalisez l’optimisation d’une nouvelle machine à vecteur de support linéaire mais en utilisant tous les attributs du jeu de données.
# ### Le score de classification en test a-t-il augmenté ? Pourquoi ?

# In[274]:


data = df[['Glucose' ,'BMI']]
y = np.array(df['Outcome'])
X = np.array(data)

X= np.array(data)
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state = 0)
lin_svc = svm.SVC(C=1, gamma=0.01, kernel='rbf')
lin_svc.fit(X_train, y_train)
lin_svc.score(X_test, y_test)


# En prenant tous les attributs cette fois-ci avec les paramètres trouvés avec notre précédant GridSearch on remarque que le score a pu s'améliorer légèrement.
# Ça s'explique par le fait qu'on a augmenté le nombre de dimensions de notre étude, on peut donc séparer les points de nos données de façon plus optimale et avec plus de facilité.
# 
# Pour aller plus loin, on a decider d'appliquer un nouveau Grid Search, ce coup-ci en prenant compte de tous les attributs du dataset, on voudra grace à ça étudier s'il y aura un changement au niveau des paramètres employés ou non et d'autre part regarder le score probable.

# In[275]:


data = df[['Pregnancies' ,'Glucose', 'BloodPressure',
           'SkinThickness' ,'Insulin','BMI', 
           'DiabetesPedigreeFunction', 'Age']]

y = np.array(df['Outcome'])
X = np.array(data)

X= np.array(data)
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state = 0)


param_grid = {'C': [0.1, 1,2,3,4, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']} 
  
grid = GridSearchCV(svm.SVC(), param_grid)
  
# fitting the model for grid search
grid.fit(X_train, y_train)


# print best parameter after tuning
print(grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

print("Score: "+str(grid.score(X_test, y_test)))


# On remarque que le GridSearch nous fournit de nouveaux paramètres et que le score s'est amélioré.
