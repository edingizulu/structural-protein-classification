## Structural Protein Classification 

### __NGIZULU__ Edi & __DIALLO__ Sadou Safa 
-----------------------------------
###  DS Formation continue  Mai 2021 ### 
------------------------------------
### I. Contexte du Projet ### 
 > Le projet fil rouge clôturant notre formation continue de data scientist chez Datascientest  porte sur
 **_la Structural Protein Classification_**. Ce projet se fixe pour objectif la prédiction de la structure des proteines avec des algorithmes de __Machine Learning__ et de __Deep Learning__.
 Le choix de ce projet qui ne fait pas partie du catalogue de projets proposé par nos formateurs a été laborieux:
 - dans la compréhension du sujet 
 - sa mise en oeuvre 
 - son interprétation des résultats etc.

> Néanmoins la volumétrie des données ainsi que les diverses catégories des variables nous ont permis de comprendre vers quels algorithmes de machine learning et de deep learning orienter notre travail et appliquer les connaissances acquises au cours de la formation.

#### Quels sont les intérêts pour la classification de la structure des protéines?

> __Les proteines__ sont des macromolécules complexes, elles sont les plus abondantes des molécules organiques des cellules et constituent à elles seules plus de 50% du poids à sec des êtres vivants. 
> L'intérêt majeur de la prédiction de la structure des proteines trouve son application en __bio-informatique, en biotechnologie et en médecine__ notamment dans la fabrication des  enzymes et de nouveaux médicaments.
Le projet communautaire __[CAMEO3D](https://www.cameo3d.org/)__  évalue les performances continues des serveurs web dédiés à la prédiction de la structure des protéines.

> __[AlphaFold](https://en.wikipedia.org/wiki/AlphaFold)__  est une IA developpée par google pour la prédiction de la structure des proteines et cette application d'intélligence artificielle a notamment servi dans la prédiction de la structure des proteines du __SARS-COV-2__
  
> _La prédiction  de la structure des protéines est l'inférence de la structure tridimensionnelle d'une protéine à partir de sa séquence d'acides aminés c'est-à-dire la prédiction de son pliage et de sa structure secondaire et tertiaire de sa structure primaire_ __[wikipedia](https://fr.wikipedia.org/wiki/Pr%C3%A9diction_de_la_structure_des_prot%C3%A9ines)__

 ### II. Source des données  
---
> Les données sont issues de la __[Protein Data Bank](https://www.rcsb.org/)__ (PDB). 
 Il s'agit d'un ensemble de données extrait de la base du groupe d'experts du Research Collaboratory for Structural Bioinformatics.
 Dans cette base  sont stockées les coordonnées atomiques des proteines et des informations rélatives à d'autres macro-molécules.
 > Le dataset est composé des deux fichiers: 
 > 
 **- protéine** : _regroupe toutes les variables décrivant la protéine_ 
 
 **- séquences :** _regroupe toutes les variables décrivant les séquences associées aux protéines_ 
 
 #### 1.  Dataset proteins
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

#### 2. Dataset sequence
|Features               |    Types         |    Description              |
|:----------------------|-----------------:|------------------------------------------------:|          
|**residueCount**       | float64          | nombre d'acides aminés dans la séquence         |
|**macromoleculeType**  | object           | Type de macromolécule (Protein,DNA, RNA)        |
|**structureId**        | float64          | id de la structure                              |
|chainId                | float64          | id de la sequence                               |
|sequence               | object           | sequence de la proteine                         |

>3 variables communes des 2 tables ont permis la fusion en une unique table. 

## III. Nettoyage des données
> Après la fusion des datasets, nous avons constaté un nombre non négligéable des
données manquantes sur certaines variables. Ainsi pour garder un maximum des données,
nous avons appliqué la stratégie de gestion des données manquantes suivante:
- remplacement par la médiane des variables numériques
- remplacement par la mode  des variables catégorielles

> Nous avons supprimé les variables n'ayant aucun impact sur l'analyse des données :

_pdbxDetails ----> diverses propriétés de la proteine sans impact sur l'analyse_

_chainId     ----> id sans impact sur l'analyse des données_

_structureId ----> id sans impact sur l'analyse des données_

> Nous avons également supprimé la variable __sequence__ pour les algorithmes de Machine
Learning; cette variable servira d'input feature dans le Deep Learning en deuxième
partie de ce projet.

Les variables quantitatives telles que _residueCount, structureMolecularWeight, densityMattews, resolution_ étaient fortement assymétriques, une correction logarithmique a été opérée.
#### Target Feature
> La variable classification est notre variable cible (target feature),
elle comprend __4989 modalités !__ . 
Ne pouvant pour des raisons pratiques faire la
classification de toutes ces modalités, dans le modele final nous n'avons gardé que
__les classes de fréquence supérieure à 5000 valeurs__ 

Pour terminer, toutes les données manquantes restantes ont été supprimées. 

Cette stratégie nous a permis d'avoir un dataset final propre et prêt pour l'analyse des données avec les __algorithmes ML__ et le __Deep Learning__.
