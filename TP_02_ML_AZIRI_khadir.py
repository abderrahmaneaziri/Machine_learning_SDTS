#!/usr/bin/env python
# coding: utf-8

# # TP : Les arbres de décision sur Car Evaluation Data Set
# 
# #### AZIRI Abderrahmane
# #### KHADIR Nadir

# ## Classification des données CAR :
# 
# On crée un chemin vers le fichier de données. 
# Ensuite, on les lit de ce fichier en utilisant pd.read_csv().
# Le DataFrame est alors créé avec toutes ses colonnes définies sur leurs features (buying=0, maint=1, doors=2, persons=3, lug_boot=4, safety = 5).
# par la suite on analyse chaque colonne individuellement et on remplace toutes les données catégoriques par des données numériques choisises au préalable.
# 
# #### Question 1: (Calculer les statistiques (moyenne et écart-type) des quatre variables explicatives)
# On affiche le resumé de la description statistique de notre jeu de données pour mieux appréhender le dataset.

# In[182]:


import pandas as pd
import numpy as np
path = r"/home/etudiant/Téléchargements/car.data"

data=pd.read_csv(path)
data=pd.DataFrame(data=data)
data.columns=['buying','maint','doors','persons','lug_boot','safety','class']


    


# #### Question 2: Combien y a-t-il d’exemples de chaque classe ?

# In[183]:


les_classes = np.unique(data['class'])

print ('On a {} classes dans notre dataset '.format(len(les_classes)))
print ('Nos classes :', les_classes)


# ##### Pour la suite, on va modifier nos données afin de pouvoir les utiliser pour l'apprentissage dans notre classifieur

# On remplace chaque valeur de chaque colonne afin d'éviter toute ambiguïté par la suite.
# 
# Les donées seront stockées dans X et les labels dans y.

# In[213]:



#pour la 1erre colonne
data['buying']=data['buying'].replace('low',0)
data['buying']=data['buying'].replace('med',1)
data['buying']=data['buying'].replace('high',2)
data['buying']=data['buying'].replace('vhigh',3)

data['maint']=data['maint'].replace('low',0)
data['maint']=data['maint'].replace('med',1)
data['maint']=data['maint'].replace('high',2)
data['maint']=data['maint'].replace('vhigh',3)

data['persons']=data['persons'].replace('more',5)
data['persons']=data['persons'].replace('2',2)
data['persons']=data['persons'].replace('4',4)

data['doors']=data['doors'].replace('5more',5)
data['doors']=data['doors'].replace('2',2)
data['doors']=data['doors'].replace('3',3)
data['doors']=data['doors'].replace('4',4)

data['lug_boot']=data['lug_boot'].replace('small',1)
data['lug_boot']=data['lug_boot'].replace('med',2)
data['lug_boot']=data['lug_boot'].replace('big',3)

data['safety']=data['safety'].replace('low',1)
data['safety']=data['safety'].replace('med',2)
data['safety']=data['safety'].replace('high',3)


data['class']=data['class'].replace('acc',0)
data['class']=data['class'].replace('good',1)
data['class']=data['class'].replace('unacc',2)
data['class']=data['class'].replace('vgood',3)

#labels=np.unique(data['class'])
y=data['class']
X = data[['buying','maint','doors','persons','lug_boot','safety']]
print(data.dtypes)
print('\n',data.describe())


# En affichant la description on remarque que les types de variables dans le dataset sont tous numériques.
# 

# ##### L'étape suivante consiste en un split de nos données afin de faire notre apprentissage et ensuite de tester notre modèle.

# In[193]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,random_state=0)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)


# * On affiche la représentation de notre arbre de décision : 

# In[191]:


fig = plt.figure(figsize=(25,20))
tree.plot_tree(clf, filled=True)


# Ensuite on applique notre modèle entrainé sur les données de test: 

# In[169]:


clf.predict(X_test)
clf.score(X_test,y_test)


# In[195]:


from sklearn.model_selection import GridSearchCV



tree_param = [{'criterion': ['entropy', 'gini'],
               'max_depth': range(1,200,5),
              'min_samples_leaf': range(1,50,2)}]



classifier = GridSearchCV(tree.DecisionTreeClassifier(), tree_param, cv=5)
classifier.fit(X_train, y_train)
print(classifier.best_params_)

classifier.predict(X_test)
classifier.score(X_test,y_test)


# #### Question 3 : Changez les valeurs de parametres max_depth et min_samples_leaf. Que constatez- vous ?
# 
# max_depth :
# 
# La profondeur maximale théorique qu'un arbre de décision peut atteindre est inférieure d'une unité au nombre d'échantillons d'apprentissage, mais aucun algorithme ne vous laissera atteindre ce point pour des raisons évidentes, l'une des principales étant l'Overfiting.
# 
# En général, plus notre arbre est profond, plus notre modèle devient complexe, car il comporte plus de divisions et capture plus d'informations sur les données. C'est l'une des causes principales de l'aOverfitting des arbres décisionnels, car notre modèle s'adaptera parfaitement aux données d'apprentissage et ne sera pas en mesure de généraliser correctement sur l'ensemble de test. Donc, si notre modèle s'adapte trop, réduire le nombre de max_depth est une façon de combattre l'adaptation excessive.
# 
# Il est également mauvais d'avoir une profondeur très faible parce que notre modèle va sous-adapter la manière de trouver la meilleure valeur, il n'y a pas une valeur unique pour toutes les solutions. Donc, ce qu'on a fais , c'est de laisser le modèle décider de la profondeur maximale d'abord et ensuite, en comparant nos résultats de train et de test, on regarde s'il y a sur- ou sous-adaptation et, selon le degré, on diminue ou augmente la profondeur maximale.
# 
# 
# min_samples_leaf:
# 
# min_samples_leaf est le nombre minimum d'échantillons requis pour être à un noeud leaf. Ce paramètre est similaire à min_samples_splits, cependant, il décrit le nombre minimum d'échantillons des échantillons aux leaf, la base de l'arbre.
# 
# Des valeurs plus élevées empêchent un modèle d'apprendre des relations qui pourraient être très spécifiques à l'échantillon particulier sélectionné pour un arbre. Des valeurs trop élevées peuvent également conduire à un sous-ajustement, c'est pourquoi nous avons ajuster les valeurs de min_samples_split en fonction du niveau de sous-ajustement ou de surajustement. L'augmentation de cette valeur peut entraîner un sous-ajustement.
# 

# #### Question : 
# #### Le problème ici étant particulièrement simple, refaites une division apprentissage/test avec 5% des données en apprentissage et 95% test. 
# #### Calculez le taux d’éléments mal classifiés sur l’ensemble de test. 
# #### Faites varier (ou mieux, réalisez une recherche par grille avec GridSearchCV) les valeurs des paramètres max_depth et min_samples_leaf pour mesurer leur impact sur ce score.

# In[205]:



# une division apprentissage/test avec 5% des données en apprentissage et 95% test. 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05,
random_state=1)



# Grid search sur les valeurs des paramètres max_depth et min_samples_leaf pour mesurer leur impact sur ce score.

tree_param = [{'max_depth': range(1,200,5),
              'min_samples_leaf': range(1,50,2)}]

classifier = GridSearchCV(tree.DecisionTreeClassifier(), tree_param, cv=5)
classifier.fit(X_train, y_train)
print(classifier.best_params_)

classifier.predict(X_test)
score = classifier.score(X_test,y_test)



# Calcule du taux d’éléments mal classifiés sur l’ensemble de test. 
print ('Le taux d\'éléments mal classifiées est : ', 1-score)


# ## Affichage de la surface de décision
# 
# #### Question :
# #### Refaire l’affichage pour les autres paires d’attributs. Sur quelle paire la séparation entre les classes est la plus marquée ?
# 
# On met en place une méthode pour pouvoir faire une combinaison de chaque deux features distinctes
# 
# #### On remarque que dans notre cas la surface de decision ne sert pas à grand chose 
# Cela est dû au fait que nos valeurs qui remplacent les valeurs catégoriques sont discrétes.i.e. les valeurs sont superposées lors de l'affichage (0,1, 2...) pour toutes les colonnes.

# #### On a affiché chaque colonne en fonction de l'autre grace à une boucle. 

# In[221]:


import numpy as np
import matplotlib.pyplot as plt

# Paramètres
n_classes = 4
plot_colors = "bryw" # blue-red-yellow
plot_step = 0.02

f = 5


while f>0: 
    j = f 
    while j>0 :
        j = j-1

        pair= [f, j]
        print(f,j)
        Xr= np.array(X)
        Xr = Xr[:,pair]

        # Apprentissage de l'arbre
        clf = tree.DecisionTreeClassifier().fit(Xr, y)
        # Affichage de la surface de décision
        x_min, x_max = Xr[:, 0].min() - 1, Xr[:, 0].max() + 1
        y_min, y_max = Xr[:, 1].min() - 1, Xr[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min,y_max, plot_step))

        classe = ['acc', 'good', 'unacc','vgood']

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        plt.xlabel(data.columns[pair[0]])
        plt.ylabel(data.columns[pair[1]])
        plt.axis("tight")
        # Affichage des points d'apprentissage
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(Xr[idx, 0], Xr[idx, 1], c=color, label=classe[i],cmap=plt.cm.Paired)
        plt.axis("tight")
        plt.suptitle("Decision surface of a decision tree using paired features")
        plt.legend()
        plt.show()
        
    f = f-1


# L'arbre de décision est une technique d'apprentissage supervisé qui peut être utilisée pour les problèmes de classification et de régression, mais elle est surtout utilisée pour résoudre les problèmes de classification. Il s'agit d'un classificateur structuré en arbre, où les nœuds internes représentent les caractéristiques d'un ensemble de données, les branches représentent les règles de décision et chaque nœud feuille représente le résultat.
# C'est appelé arbre de décision parce que, comme un arbre, il commence par le nœud racine, qui se développe sur d'autres branches et construit une structure arborescente.

# L'arbre crée une représentation visuelle de tous les résultats possibles, des récompenses et des décisions de suivi dans un seul document. Chaque décision ultérieure résultant du choix initial est également représentée sur l'arbre, de sorte que nous pouvons voir l'effet global de toute décision. En parcourant l'arbre et en faisant des choix, on voit un chemin spécifique d'un nœud à l'autre et l'impact qu'une décision prise maintenant pourrait avoir plus tard.
