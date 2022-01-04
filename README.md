# Structural Protein Classification 




**README**
 <center> <h3> NGIZULU Edi & DIALLO Sadou Safa </h3> <center>\n
                
        <center> <h3> Structural Protein Classification </h3> <center>\n
    
        <center> <h3> Formation continue DataScientiste Mai 2021 </h3> <center>\n
                
              \n
              \n
              <br>
              <br>
              
        <center> <h2> Contexte </h2> <center>\n
             > Le projet fil rouge clôturant notre formation continue de data scientiste chez Datascientest  porte sur  la __Structural Protein Classification__
             ><br> <br>
             > Le choix de ce projet qui est hors catalogue a été laborieux:
                 <li> dans la compréhension du sujet et sa mise en oeuvre
                 <li> dans l'interprétation des résultats 
             > <br> <br>
             > Néanmoins la volumétrie des données ainsi que les diverses catégories des variables nous ont permis de comprendre vers quels algorithmes de machine learning et de deep learning orienter  nos recherches et appliquer les connaissances acquises au cours de la formation.
             > <br> <br>
             > En effet, __quels intérêts pour la classification de la structure des proteines ?__
             > __Les proteines__ sont des macromolécules complexes, elles sont les plus abondantes des molécules organiques des cellules et constituent à elles seules plus de 50% du poids\n à sec des êtres vivants. <br>
             > <br> <br> 
             > La prédiction de la structure des protéiques est une préoccupation majeure en bio-informatique, en biotechnologie et en médecine notamment dans la conception des enzymes et des nouveaux médicaments
             > <br> <br> 
             > Le projet communautaire [CAMEO3D](https://www.cameo3d.org/) évalue les performances continues des serveurs web dédiés à la prédiction de la structure des protéines  
             > 
## Presentation

This repository contains the code for our project **Structural Protein Classification**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).
The goal of this project is to **predict the structure of proteins with Machine Learning and Deep Learning algorithms.**

This project was developed by the following team :

- Edi __NGIZULU__ ([GitHub](https://github.com/) / [linkedin](www.linkedin.com/in/edi-ngizulu-57256316a))
- Sadou Safa __DIALLO__ ([GitHub](https://github.com/) /[linkedin](https://www.linkedin.com/in/sadou-safa-diallo-a0839b49/))

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```

## Streamlit App
This project aims to predict the structure of proteins with Machine Learning and Deep Learning algorithms.

**Add explanations on how to use the app.**

To run the app :

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

**Docker**

You can also run the Streamlit app in a [Docker](https://www.docker.com/) container. To do so, you will first need to build the Docker image :

```shell
cd streamlit_app
docker build -t streamlit-app .
```

You can then run the container using :

```shell
docker run --name streamlit-app -p 8501:8501 streamlit-app
```

And again, the app should then be available at [localhost:8501](http://localhost:8501).
