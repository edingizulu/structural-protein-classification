# Structural Protein Classification 




**README**

## Presentation

This repository contains the code for our project **Structural Protein Classification**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).
The goal of this project is to **predict the structure of proteins with Machine Learning and Deep Learning algorithms.**

This project was developed by the following team :

- Edi __NGIZULU__ ([GitHub](https://github.com/) / [linkedin](www.linkedin.com/in/edi-ngizulu-57256316a)
- Sadou Safa __DIALLO__ ([GitHub](https://github.com/) /[linkedin](https://www.linkedin.com/in/sadou-safa-diallo-a0839b49/)

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
