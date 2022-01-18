![protein_structure](https://user-images.githubusercontent.com/35880186/149831530-3020edf8-5c2b-446f-be2b-b9f16904abcc.jpeg)


## Structural Protein Classification 

### __NGIZULU__ Edi & __DIALLO__ Sadou Safa 
-----------------------------------
###  DS Formation continue  Mai 2021 ### 
------------------------------------
###  Contexte du Projet ### 
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

 ### I. ANALYSE DES DONNEES 
   ---
#### 1. Source de données  
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
  ####  2.  Nettoyage des données
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

####  2. Analyse Exploratoire des données 
---
 ##### Classification
---
La variable classification est notre variable cible, elle comprend 4989 modalités. Nous avons restreint les classes à prédire à 17 correspondant aux classes ayant 
une fréquence supérieure à 5000 valeures.
![target_50_most](https://user-images.githubusercontent.com/35880186/149831023-617bbf5a-ec54-495b-b8a5-3a22a5c8b24f.PNG)

![countplot_target](https://user-images.githubusercontent.com/35880186/149831302-42efc083-28c6-4df6-848f-ab69ae03d6cf.PNG)

##### Technique d'extraction de la Proteine
---
> Avant de séquencer la proteine, son extraction s'obtient par plusieurs types de technique (32 techniques) dont la plus utilisée est la __X-RAY-DIFFRACTION__
répresentant à elle seule __86 %__ des techniques utilisées dans la base, deux autres techniques s'ajoutent à celle-ci formant __99%__ des solutions techniques utilisées dans le dataset.

> Ce constat nous a amené à restreindre le dataset final à ces 3 techniques et regrouper les autres dans la modalité "other_techniques".
Cette étape est importante car elle permet d'obtenir la séquence de la proteine et donc de la classification de sa structure lui affectant un rôle dans la cellule. 

![pie_technique_protein_extraction](https://user-images.githubusercontent.com/35880186/149831861-43c30cc6-56d1-4ab7-b92d-27e3eaab2798.png)

 ##### Valeur du Ph
---
> Les liaisons protéiques peuvent être changées voire disloquées par des agents de dénaturation tels que le PH (potentiel hydrogène). Globalement les protéines de la base sont plus neutres que basiques(supérieur à 7 ). Leur acidité(inférieur à 7) étant à 
cheval entre les deux prémières. Cette variable a été recodée en variable catégorielle. 

![phvalue](https://user-images.githubusercontent.com/35880186/149834550-58b4efa9-c960-4a87-bc4c-00b4aceb3295.PNG)

![count_ph_target](https://user-images.githubusercontent.com/35880186/149834649-af75b527-ef05-4f29-b71b-68fa4b7de00f.PNG)

>> _Nous n'observons pas de tendance dans la distribution des classes de proteine selon les valeurs du PH à part la classe __ribosome__ qui ressort beaucoup plus neutre, __l'hydrolase__ plus acidulé_

 ##### Type de Macromolécule
---
> Nous avons environ **80%** du dataset qui est composé de protéine, 18% composé de protéines avec ARN ou ADN. Nous avons fait le choix de garder ces 3 modalités. 
> 
![count_macromolecule_target](https://user-images.githubusercontent.com/35880186/149835733-721ba08b-b31c-4665-93f3-622d220897cb.png)

>> _Comme attendu nous constatons plus de protéine dans le dataset final_

##### Méthode de cristallisation
--- 
Nous avons dénombrés 418 techniques de cristallisation parmi lesquelles 4 à elles seules répresentent 94% du dataset total

![methodes_crystallizations](https://user-images.githubusercontent.com/35880186/149836216-9c5a95e9-d8b7-44dd-8447-f969ab81b9b6.PNG)

![count_crystalization_target](https://user-images.githubusercontent.com/35880186/149836141-d8ee744b-5ea8-44b2-b26f-49e567735292.PNG)

>> _Les méthodes de vaportisations sont les plus répresentées dans le dataset final_ 
