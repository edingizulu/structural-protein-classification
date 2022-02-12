
# Structural Protein Classification 

![protein_structure](https://user-images.githubusercontent.com/35880186/149831530-3020edf8-5c2b-446f-be2b-b9f16904abcc.jpeg)
---
### __NGIZULU__ Edi      
_[linkedin](www.linkedin.com/in/edi-ngizulu-57256316a)_
### __DIALLO__ Sadou Safa  
_[linkedin](https://www.linkedin.com/in/sadou-safa-diallo-a0839b49/)_

-----------------------------------
###  DS Formation continue  Mai 2021 
------------------------------------

### Sommaire
[Contexte du projet](#contexte)

#### I. [Analyse des données](#analyse)
   1. [Source des données](#source)
   2. [Nettoyage des données](#nettoyage)
   3. [Analyse Exploratoire des données ](#eda)
   
       - [Classification](#target)
       - [Technique d'extraction de la Proteine](#technique)
       - [PH Value](#ph)
       - [macromoleculeType](#macro)
       - [methodes_crystallizations](#cristal)
       - [residueCount](#residue)
       - [structureMolecularWeight](#weight)
       - [publicationYear](#pub)
       - [Sequence Feature](#seq)
#### A-[MACHINE LEARNING](#ml)     
#### II. [Modélisation](#modelisation)
   1. [Preprocessing](#preprocessing)
   2. [Métriques des tests](#metric)
   3. [Itération des Modèles](#iteration)
   
       3.1 [Itération 1: Lazypredict()](#iteration1)
       
       3.2 [Itération 2: Performances prédictives](#iteration2)
       
       3.3 [Itération 3: Modèles retenus](#modeles)
       
   4. [Métrique des modèles](#metriques)
   
   5. [Optimisation des paramètres: Tuning](#tuning)
   6. [Interprétatbilité du Modèle](#retainmode)
  
 #### B- [DEEP LEARNING](#deep)
 #### III-[Deep Modélisation](#deepmodel)
 
   1. [Convolutional Neural Network](#cnn)
   
   2. [Résultats](#deepresult)
---
###  Contexte du Projet  <a name="contexte"></a>
---
 > Le projet fil rouge clôturant notre formation continue de data scientiste chez Datascientest  porte sur
 **_la Structural Protein Classification_**. Ce projet se fixe pour objectif la prédiction de la structure des proteines avec des algorithmes de __Machine Learning__ et de Deep __Learning__.
 Le choix de ce projet qui ne fait pas partie du catalogue de projets proposé par nos formateurs a été laborieux:
 - dans la compréhension du sujet 
 - sa mise en oeuvre 
 - son interprétation des résultats etc.

> Néanmoins la volumétrie des données ainsi que les diverses catégories des variables nous ont permis de comprendre vers quels algorithmes de machine learning et de deep learning orienter  nos recherches et appliquer les connaissances acquises au cours de la formation.

##### Quels sont les intérêts pour la classification de la structure des protéines?

> __Les proteines__ sont des macromolécules complexes, elles sont les plus abondantes des molécules organiques des cellules et constituent à elles seules plus de 50% du poids à sec des êtres vivants. 
> L'intérêt majeur de la prédiction de la structure des proteines trouve son application en __bio-informatique, en biotechnologie et en médecine__ notamment dans la fabrication des  enzymes et de nouveaux médicaments.
Le projet communautaire __[CAMEO3D](https://www.cameo3d.org/)__  évalue les performances continues des serveurs web dédiés à la prédiction de la structure des protéines.

> __[AlphaFold](https://en.wikipedia.org/wiki/AlphaFold)__  est une IA developpée par google pour la prédiction de la structure des proteines et cette application d'intélligence artificielle a notamment servi dans la prédiction de la structure des proteines du __SARS-COV-2__
  
> _La prédiction  de la structure des protéines est l'inférence de la structure tridimensionnelle d'une protéine à partir de sa séquence d'acides aminés c'est-à-dire la prédiction de son pliage et de sa structure secondaire et tertiaire de sa structure primaire_ __[wikipedia](https://fr.wikipedia.org/wiki/Pr%C3%A9diction_de_la_structure_des_prot%C3%A9ines)__

 ### I. ANALYSE DES DONNEES <a name="analyse"></a>
   ---
### 1. Source des données  <a name="source"></a>
---
> Les données sont issues de la __[Protein Data Bank](https://www.rcsb.org/)__ (PDB). 
 Il s'agit d'un ensemble de données extrait de la base du groupe d'experts du Research Collaboratory for Structural Bioinformatics.
 Dans cette base  sont stockées les coordonnées atomiques des proteines et des informations rélatives à d'autres macro-molécules.
 > Le dataset est composé des deux fichiers: 
 >> **_protéine_ :**  regroupe toutes les variables décrivant les propriétés physiques des amino-acides (protéines) 
 >> 
 >> **_séquences_:**  regroupe toutes les variables décrivant les séquences associées à chaque acide aminé (protéine) 
 
 ###### 1.  Dataset proteins
|Features               |    Types         |         Description              |
|:----------------------|-----------------:|------------------------------------------------:|
| Classification        | object           | classification de la molécule                   |  
|  crystallizationMethod| object           | Méthode de cristallisation de la protéine       |
| crystallizationTempK  | float64          | température à laquelle la protéine cristallise  |
| densityMatthews       | float64          | volume cristallin par unité de poids moléculaire|
| densityPercentSo      | float64          | % de la densité du solvant dans la proteine     |
| experimentalTechnique | object           | Méthode d'obtention de  la structure protéique  |  
| pdbxDetails           | object           | divers détails sur la proteine                  |
| phValue               | object           | Ph de la solution (acide, basique,neutre)       |
|publicationYear        | float64          | année de publication de la structure protéique  |
|**residueCount**       | float64          | nombre d'acides aminés dans la séquence         |
|resolution             | float64          | qualité du cristal contenant la protéine        |
|structureMolecularWeight|float64          | masse moléculaire en kilo dalton                |
|**structureId**        | float64          | id de la strcuture                              |
|**macromoleculeType**  | object           | Type de macromolécule (Protein,DNA, RNA)        |   

> 

###### 2. Dataset sequence
|Features               |    Types         |    Description              |
|:----------------------|-----------------:|------------------------------------------------:|          
|**residueCount**       | float64          | nombre d'acides aminés dans la séquence         |
|**macromoleculeType**  | object           | Type de macromolécule (Protein,DNA, RNA)        |
|**structureId**        | float64          | id de la structure                              |
|chainId                | float64          | id de la sequence                               |
|sequence               | object           | sequence de la proteine                         |

>3 variables communes des 2 tables ont permis la fusion en un tableau unique  
 ---
  ####  2.  Nettoyage des données   <a name="nettoyage"></a>
 ---
> Après la fusion des datasets, nous avons constaté un nombre non négligéable des
données manquantes sur certaines variables. Ainsi pour garder un maximum des données,
nous avons appliquer la stratégie de gestion des données manquantes suivante:
- remplacement par la médiane des variables numériques
- remplacement par la mode  des variables catégorielles

> Nous avons supprimé les variables n'ayant aucun impact sur l'analyse des données :

_pdbxDetails ----> diverses propriétés de la proteine sans impact sur l'analyse_

_chainId ----> id sans impact sur l'analyse des données_

_structureId ----> id sans impact sur l'analyse des données_

> Nous avons également supprimé la variable __sequence__ pour __la partie Machine
Learning__ et nous l'avons gardé comme input feature en deuxième partie de ce projet consacré au __Deep Learning__.

Les variables quantitatives telles que _residueCount, structureMolecularWeight, densityMattews, resolution_ étaient fortement assymétriques, une correction logarithmique a été opérée.

##### Target Feature
> La variable classification est notre variable cible (target feature),
elle comprend __4989 modalités !__ . 
Ne pouvant pour des raisons pratiques faire la
classification de toutes ces modalités, dans le modele final nous n'avons gardé que
__les classes de fréquence supérieure à 5000 valeurs__ 

Pour terminer, toutes les données manquantes restantes ont été supprimées. 

Cette stratégie nous a permis d'avoir un dataset final propre et prêt pour l'analyse des données avec les __algorithmes ML__ et le __DL__.

####  3. Analyse Exploratoire des données <a name="eda"></a>
---
 - ##### Classification <a name="target"></a>
---
La variable classification est notre variable cible, elle comprend 4989 modalités. Nous avons restreint les classes à prédire à 17 correspondant aux classes ayant 
une fréquence supérieure à 5000 valeures.
![target_50_most](https://user-images.githubusercontent.com/35880186/149831023-617bbf5a-ec54-495b-b8a5-3a22a5c8b24f.PNG)

![countplot_target](https://user-images.githubusercontent.com/35880186/150330112-f8bdf880-675f-4610-8183-c8fbf9cb8732.PNG)


---
- ##### Technique d'extraction de la Proteine <a name="technique"></a>
---
> Avant de séquencer la proteine, son extraction s'obtient par plusieurs types de technique (32 techniques) dont la plus utilisée est la __X-RAY-DIFFRACTION__
répresentant à elle seule __86 %__ des techniques utilisées dans la base, deux autres techniques s'ajoutent à celle-ci formant __99%__ des solutions techniques utilisées dans le dataset.

> Ce constat nous a amené à restreindre le dataset final à ces 3 techniques et regrouper les autres dans la modalité "other_techniques".
Cette étape est importante car elle permet d'obtenir la séquence de la proteine et donc de la classification de sa structure lui affectant un rôle dans la cellule. 

![pie_technique_protein_extraction](https://user-images.githubusercontent.com/35880186/149831861-43c30cc6-56d1-4ab7-b92d-27e3eaab2798.png)
---
 - ##### Valeur du Ph <a name="ph"></a>
---
> Les liaisons protéiques peuvent être changées voire disloquées par des agents de dénaturation tels que le PH (potentiel hydrogène). Globalement les protéines de la base sont plus neutres que basiques(supérieur à 7 ). Leur acidité(inférieur à 7) étant à 
cheval entre les deux prémières. Cette variable a été recodée en variable catégorielle. 

![phvalue](https://user-images.githubusercontent.com/35880186/149834550-58b4efa9-c960-4a87-bc4c-00b4aceb3295.PNG)

![count_ph_target](https://user-images.githubusercontent.com/35880186/149834649-af75b527-ef05-4f29-b71b-68fa4b7de00f.PNG)

>> _Nous n'observons pas de tendance dans la distribution des classes de proteine selon les valeurs du PH à part la classe __ribosome__ qui ressort beaucoup plus neutre, __l'hydrolase__ plus acidulé_
>>
---
 - ##### Type de Macromolécule <a name = "macro"></a>
---
> Nous avons environ **80%** du dataset qui est composé de protéine, 18% composé de protéines avec ARN ou ADN. Nous avons fait le choix de garder ces 3 modalités. 
> 
![count_macromolecule_target](https://user-images.githubusercontent.com/35880186/149835733-721ba08b-b31c-4665-93f3-622d220897cb.png)

>> _Comme attendu nous constatons plus de protéine dans le dataset final_
>> 
---
- ##### Méthode de cristallisation <a name = "cristal"></a>
--- 
Nous avons dénombrés 418 techniques de cristallisation parmi lesquelles 4 à elles seules répresentent 94% du dataset total

![methodes_crystallizations](https://user-images.githubusercontent.com/35880186/149836216-9c5a95e9-d8b7-44dd-8447-f969ab81b9b6.PNG)

![count_crystalization_target](https://user-images.githubusercontent.com/35880186/149836141-d8ee744b-5ea8-44b2-b26f-49e567735292.PNG)

>>  Les méthodes de vaporisation comptent pour 90% du dataset, le microbatch 3%. Les techniques de cristallisation par la vapeur sont comme attendues les plus fréquentes dans le dataset final. 

---
- #### residueCount par classe de protéine <a name = "residue"></a>
---
> Nous avons comparé les 17 classes de protéine selon le nombre d'acides aminés (feature residueCount), nous n'observons pas des différences particulières entre les classes 
hormis les classes __virus__, __ribosome__ et __ribosome/antibiotic__

![box_residuecount](https://user-images.githubusercontent.com/35880186/150307855-4d06370e-e7f2-4dee-9035-4cc229d9ec2c.PNG)

---
- #### structureMolecularWeight par classe de protéine<a name = "weight"></a>
---
> Nous avons fait aussi la comparaison des classes de protéine avec le poids de la structure moléculaire,  comme précédemment nous n'observons pas des différences particulières entre les classes,  les classes précédentes ressortent( __virus__, __ribosome__ et __ribosome/antibiotic__) montrant une corrélation entre ces deux variables sur ces 3 classes.
![box_macromolecule](https://user-images.githubusercontent.com/35880186/150309148-7f1ced98-6997-4828-a685-61a622f4fcd3.PNG)

---
 - #### publicationYear par classe de protéine <a name="pub"></a>
---
> Cette variable bien qu'intervenant peu dans la prédiction de la structure des protéines nous renseigne sur l'intense activité de la communauté des chercheurs du RCB et 
l'intérêt grandissant de la thématique des protéines depuis 2015 (année médiane)

![box_publication](https://user-images.githubusercontent.com/35880186/150310358-6c7ffeaa-de25-4590-bc2d-2ddf4fdccd0b.PNG)

----
- #### Sequence Feature <a name="seq"></a>
---
> Notre objectif en deuxième partie de ce projet étant le deep learning, la variable sequence sera utilisée comme seule variable explicative; elle est composée de 25 lettres de 
> longueur différente 
> 
![histplot_seq_freq](https://user-images.githubusercontent.com/35880186/151719520-77c9fa26-4c51-483b-bff0-91b61288f2f2.png)

>> Les lettres __G__ et __A__ sont les fréquentes.

![seq_frequences_barplot](https://user-images.githubusercontent.com/35880186/151719278-89e8faf8-759f-4aa8-bc52-ac6bce7c493c.png)

---
## A. MACHINE LEARNING  <a name ="ml"></a>
>> Nous avons fait le choix d'adopter une double démarche dans l'analyse des données : 
>> - la prédiction de la structure des protéines avec les algorithmes de __Machine Learning__ par l'analyse des caractéristiques physiques des protéines
>> - la prédiction de la structure des protéines avec le __Deep Learning__ en nous basant uniquement cette fois sur l'analyse des séquences protéiques 
---
## II. Modélisation <a name ="modelisation"></a>
---
> Le but de notre projet comme rappelé précédemment est la prédiction de la structure des protéines. Nous avons choisi d'utiliser les algorithmes de ML et DL, pour ce faire, nous avons abordé les étapes ci-dessous qui nous ont permis de modéliser les données et obtenir un dataset final propre et adapté à l'entrainement des différents modèles.
### 1. Preprocessing des données:  <a name ="preprocessing"></a>
---
> L'exploration et la phase d'analyse terminées, nous avons nettoyé  et préparé le jeu de données pour l'apprentissage. 
Cette étape nous a permis:
  - d'identifier les variables non indispensables dans le dataset final, variables que nous avons supprimées
  - de gérer les données manquantes en les remplaçant par la médiane pour les variables numériques, la mode pour les variables catégorielles et supprimer les lignes des variables pour lesquelles les données étaient manquantes.
  - de réduire les modalités de certaines variables en les régroupant selon le nombre d'observation dans une nouvelle modalité.
  - de définir une stratégie de réduction des classes de protéines à prédire en les ramènant à 17 classes au lieu de 4989  en ne tenant compte que des modalités ayant plus de 5000 observations. Nous avons ainsi gardé 64% des classes protéiques dans le dataset final. 
  - de discrétiser les variables catégorielles à plusieurs modalités. 
  - Nous avons procédé par élimination récursive des variables (RFE) selon leur poids en utilisant l'algorithme des forêts aléatoires en ne gardant que 8 variables dans le dataset final. 
>> Les diverses étapes listées ci-dessus nous ont permis d'avoir un jeu données final de __310.000 lignes__ (__68%__ du dataset initial) et __8 features__ 

### 2. Métriques des tests <a name ="metric"></a>
---
> Nous avons utilisé les métriques suivantes pour la classification de la structure protéique:
>> __Accuracy:__ cette métrique nous a permis d'obtenir rapidement la performance de nos modèles 
>> 
>> __Rapport de classification:__ les performances fines sur chaque classe protéique ont été obtenues avec cette métrique 
>> 
>> __Matrices de confusion__: en détails, cette métrique nous a permis de comprendre les classifications correctes et incorrectes de des modèles. 

### 3. Itération des Modèles <a name = "iteration"></a>
---
> Nous avons utilisé la librairie _lazypredict_ pour gérer le choix difficile de la pléthore des algorithmes de classification existant. En effet, cette bibliothèque par sa simplicité d'utilisation avec peu de codes et sans réglage des hyperparamètres nous a permis de faire le choix des meilleurs modèles à retenir, modèles auxquels seront appliqués des paramètres d'optimisations par la suite. 
---
> -  #### 3.1 Itération 1 <a name = "iteration1"></a>
> Nous n'avons pas eu à choisir les modèles, le choix a été opéré automatiquement par le package avec des métriques de performance des différents modèles en ordre décroissant. 
![Lazy_train_score](https://user-images.githubusercontent.com/35880186/153007082-c415c98a-4ed9-4dae-ba29-dd3d719fdfcd.PNG)
>> Les algorithmes classiques de classification (regression logistique, SGDClassifier, lassoClassifier...) se sont montrés peu performants au contraire des algorithmes d'ensemble qui avec un temps d'entrainement rélativement courts affichent des métriques élèvées. 
>> Sur les 25 modèles testés 4 se sont montrés particulièment performants et seront utilisés en troisième itération avec des paramètres d'optimisation adaptés. 
![Lazy_train_score_graph](https://user-images.githubusercontent.com/35880186/153006627-d04a825c-880f-4b18-9702-8779065b5ceb.PNG)
---
> -  #### 3.2 Itération 2: Performances prédictives <a name = "iteration2"></a>
>> Pour arrêter définitivement notre choix sur les modèles retenus, nous avons testé avec le package Lazypredict les performances prédictives des 25 modèles et nous avons constaté que ces performances sont les mêmes que celles enrégistrées en apprentissage. 

![predict_train](https://user-images.githubusercontent.com/35880186/153010652-de8fac8a-bf2c-441c-a62b-3ee69015fba2.PNG)

#### - Difficultés rencontrées: 
Dans ces deux itérations, la principale difficulté rencontrée a été l'entrainement des modèles avec le package [Lazypredict](https://pypi.org/project/lazypredict/). Ce package malgré son utilité a des serieux problèmes de mise à jour qui ne facilitent pas son utilisation notamment les dépendences liées à d'autres bibliothèques. Pour sa mise en oeuvre, il nous a fallu l'installer dans un environement virtuel dédié. 

---
> -  #### 3.3 Itération 3:  Modèles retenus <a name = "modeles"></a>

> - #### Objectif  
> confirmer ou infirmer les prédictions du package Lazypredict des modèles prédéfinis précédemment. 
>> - #### Résultats 
>> Sur les 12 modèles testés, 4 se dégagent nettement avec des accuracy élèvés, ces modèles sont essentiellement des algorithmes de classification d'ensemble: 

<li> ExtraTreesClassifier, 
<li> RandomForest, 
<li> Le Bagging Classifier,
<li> DecisionTreeClassifier
   
![model_accuracy](https://user-images.githubusercontent.com/35880186/153227827-0a7709cd-f610-4726-b5d3-9e12e4ccf289.png)

 >> Le modèle __ExtraTrees__ s'est montré un plus performant que les autres modèles. 
   
- #### Problèmes
>> Les performances élèvées de nos 4 modèles retenus en terme d'accuracy suscitent de la prudence dans l'interprétation des résultats. En effet, un surapprentissage de nos modèles pourrait avoir pour effet une difficulté de généralisation de ceux-ci sur de nouveaux jeux de données en terme prédictif. 
   Un des meilleurs moyens de voir un effet de surapprentissage sur l'echantillon d'apprentissage et plus globalement sur la taille du jeu de données des modèles est la répresentation de ceux-ci en courbe d'apprentissage
   
![learning_rate_curve_baging](https://user-images.githubusercontent.com/35880186/153231381-be01cdc4-3445-4cc9-b05f-8c4400063b69.png)
![learning_rate_curve_boosting](https://user-images.githubusercontent.com/35880186/153231384-4f9c5bf3-f068-4401-b469-4627a82b0ef5.png)
![learning_rate_curve_rf](https://user-images.githubusercontent.com/35880186/153231388-0b39f435-9818-4080-9cd3-14917fa32078.png)
![learning_rate_extratrees](https://user-images.githubusercontent.com/35880186/153231392-5f1b37d3-b747-4bc4-8abe-9d5837d8e938.png)
   
__La lecture de l'allure des courbes d'apprentissage ne nous permet pas d'exclure complétement un effet d'overfitting (surapprentissage)__
 > - #### Difficultés rencontrées 
 >> Les difficultés rencontrées ont été principalement le temps d'apprentissage des modèles sur une machine de 8 Go de RAM (16 heures), ce temps a été ramèné à 4h50' dans une autre machine plus adaptée avec 16 Go de RAM
  
 #### 4. Métriques des différents modèles <a name = "metriques"></a>
   
 Sur les 4 modèles testés, nous avons ajouté un meta modèle qui est le __voting classifier__ 
 > - #### ExtraTreesClassifier
  ---
   
![classification_report_extratree](https://user-images.githubusercontent.com/35880186/153300268-3de99052-7b97-4979-8e10-a2dbdc6f93de.PNG)
![confusion_matrix_extra](https://user-images.githubusercontent.com/35880186/153300270-1a7846f3-ccc5-4ddd-8389-a43689c3d2db.png)

> - #### DecisionTreeClassifier
 ---

 ![classification_report_dt](https://user-images.githubusercontent.com/35880186/153302273-5b516a53-e2e6-477d-9ef6-7a4d00d34e53.PNG)
 ![confusion_matrix_dt](https://user-images.githubusercontent.com/35880186/153302328-b29c482a-146d-4e36-b1eb-79aa4cb6c3a9.png)

> - #### RandomForestClassifier
 ---
![classification_report_rf](https://user-images.githubusercontent.com/35880186/153302672-a2b2e137-2a9f-482f-a686-5e33b761618e.PNG)
![confusion_matrix_rf](https://user-images.githubusercontent.com/35880186/153302704-699e585c-51f1-4c71-bd14-7e29905aff9e.png)

 > - #### BaggingClassifier
 ---
   
   ![classification_report_bagging](https://user-images.githubusercontent.com/35880186/153303547-d726b66f-f542-47ad-a808-cefe7c1f1db0.PNG)
   ![confusion_matrix_bagging](https://user-images.githubusercontent.com/35880186/153303567-a16eb94a-fc85-4ca5-9609-7ff0bd0701e2.png)
   
 > - #### VotingClassifier
 ---
   ![classification_report_voting](https://user-images.githubusercontent.com/35880186/153304020-0099fe8e-26bc-4f2c-ab73-bca86a5125fe.PNG)
   ![confusion_matrix_voting](https://user-images.githubusercontent.com/35880186/153304039-11fca4d3-ab39-4dad-a3d5-b3209f4920ba.png)

   > - #### Résultats des Performances Globales des Modèles
   ---
   ![result_resume](https://user-images.githubusercontent.com/35880186/153304238-549c8f7d-cb71-4d27-9f79-6131b85a9814.PNG)
   
   ![result_resume'](https://user-images.githubusercontent.com/35880186/153304251-81bdde8d-dd5a-465d-9adb-ce04aa72dfc1.PNG)
   
   ---
 > - ### Optimisation des paramètres: Tuning <a name = "tuning"></a>
 > Le modèle qui s'est montré un peu plus performant sur les 4 modèles retenus reste __ExtraTreesClassifier__ (Extremely Randomized Trees) dont l'accuracy en echantillon test a été de __92%__. Nous avons pour la partie demo streamlit choisi ce modèle en plus de cette performance sa propension à contrôler le surapprentissage (pas totalement exclus comme vu précédemment). 
   
 >  - ### Objectif
   Optimiser les paramètres du modèle ET (ExtraTreesClassifier) avec GridSearch pour améliorer ses performances. 
 >  - ### Résultats:    
L'optimisation des paramètres n'améliore pas d'avantage le modèle qui était déjà  performant, tout au plus nous constatons pour la classe ribosome une meilleure prédiction. 
   
![ExtraTreesGrid](https://user-images.githubusercontent.com/35880186/153682403-8b51d74c-9a05-4707-8d17-4e737b5a34af.PNG)
 
![classification_report_gridsearch](https://user-images.githubusercontent.com/35880186/153682456-eec53e75-ef20-4913-99c2-4ce2abad7ea1.PNG)
   
![confusion_matrix_gridserachcv_ET](https://user-images.githubusercontent.com/35880186/153683505-43641f83-6b58-4670-a8d5-b34454a8d662.png)

   > - ### Interpretatbilité du Modèle <a name = "retainmode"></a>
L'algorithme ET s'appuie sur certaines variables importantes pour prédire les classes de protéines. Ici le poids moléculaire a toute son importance dans la prise de décision de l'algorithme, il en est de même du residuecount ou encore de la résolution. 
   
![ET_FeaturesImportance](https://user-images.githubusercontent.com/35880186/153686142-f7047720-4866-4841-b5cd-cb48011bce51.png)
---
> Nous n'avons pas pu utiliser le package shap pour une interprétation fine du modèle ExtraTrees, les shap_values n'ayant pu être extraites de la fonction explainer du module, les temps de calcul extrêment allongés (plus de 24h). Est-ce dû à la volumétrie des données? 
Les packages eli5 et lime nous ont permis d'avoir une interprétation locale de l'algorithme. 
- Eli5: 
   > Les coefficients associés à chaque variable sont de même ordre que les features importances vus précédemment.

   > 
   ![eli5_coeff](https://user-images.githubusercontent.com/35880186/153714539-9b644528-46cb-47a7-822f-06e64503232b.PNG)
- Lime: 
   > Ici avec le package la contribution de chaque variable dans la prédiction de "l'individu 1"
   
   ![lime_decision](https://user-images.githubusercontent.com/35880186/153715039-9a7c451f-253f-4420-9ef9-b9109de5ecfc.PNG)


#### DEEP LEARNING <a name = "deep"></a>
---
> Dans cette deuxième de notre projet, nous aborderons deux modèles de deep learning: un modèle convolutionnel à une dimension et en dernier le modèle d'apprentissage profond le LSTM.
   
#### Deep Modélisation <a name = "deepmodel"></a>
> Notre fichier de données ayant été nettoyé dans la modélisation précédente, nous n'avons gardé que les features __sequence__ et __target__. 
> - #### Convolutionnal Neural Network (CNN) <a name = "cnn"></a>:
   >> Les réseaux de neurone convolutionnels bien que souvent appliqués en imagerie pour la classification , peuvent aussi être utilisés dans la classification des séquences.Ici la séquence d'entrée est utilisée comme une image .  

> Comme précédemment les métriques utilisées sont les mêmes (accuracy, classification report, matrice de confusion).
Nous avons construit le modèle convolutionnel de façon séquentielle avec des couches denses de batchnormalization, de Maxpooling1D et des couches denses full connected.  

![summary_sequential_cnn1_deep](https://user-images.githubusercontent.com/35880186/153691674-90629687-0099-4279-83c8-f60849fa0c9c.png)  
   
![summary_plot_cnn1_deep](https://user-images.githubusercontent.com/35880186/153691428-2e1dc6c1-e1fc-484e-a75e-6a28edbd86ce.png)

   > - #### Résultats: <a name = "deepresult"></a>
   
   ![model_loss_accuracy_by_epoch](https://user-images.githubusercontent.com/35880186/153716413-1dc3bd1b-04d7-496d-b6fc-5d5250ffb793.png)
   
   > - ##### train accuracy
    
   ![train_accuracy_deep_cnn](https://user-images.githubusercontent.com/35880186/153716176-cd18df2f-0336-4370-b754-b690be6f2148.PNG)
   
   > - ##### test accuracy
![test_accuracy_deep_cnn](https://user-images.githubusercontent.com/35880186/153716175-46dca66f-aa8b-433e-8372-834909d2bcff.PNG)

 >> L'accuracy sur les données d'apprentissage et test sont proches et globalement le réseau convolutionnel reste moins efficace que le model ExtraTrees mais à priori il n'y a pas d'overfitting, ce qui est non négligeable dans la généralisation du modèle. 
