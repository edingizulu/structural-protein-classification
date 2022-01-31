import streamlit as st


title = "Structural Protein Classification"
sidebar_name = "Présentation"
img_dir = '../images/'

def run():

    # TODO: choose between one of these GIFs
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    st.image(img_dir+'sequence_p.png')

    st.title(title)

    st.markdown("--------")

    st.markdown(
      """
      ## Qu'est ce que c'est ? 
          Les protéines sont des macromolécules organiques présentes dans toutes les cellules vivantes. 
          Elles sont les plus abondantes des molécules organiques des cellules et constituent à elles seules 
          plus de 50% du poids à sec des êtres vivants.
      
      ## A quoi ça sert? 
          Ils remplissent de multiples fonctions pour les cellules. Elles interviennent pour des fonctions de transport (notamment d'oxygène),
          comme enzymes, ou hormones.
      """
    )
    st.image(img_dir+'proteinfunctions.png')

    st.markdown(
      """
      ## Les classifier? Pourquoi?
      Connaître la classe d'une protéine, revient à identifier sa fonction dans la cellule. Il est donc capital de connaître sa composition (séquence),
      ses propriétés physiques, dans le but de savoir à quoi elle sert dans la cellule.
      """
    )

    st.info("""
      Le but de notre projet est d'étudier la prédiction de la classe d'une protéine, en nous basant sur les données fournies par la Protein Data Bank
      Ces données comprennent à la fois la séquence complète de la macromolécule, ainsi que ses propriétés physiques et les méthodes utilisées pour 
      les obtenir. 
    """ )
    #st.markdown(
    #    """
    #    Here is a bootsrap template for your DataScientest project, built with [Streamlit](https://streamlit.io).

    #    You can browse streamlit documentation and demos to get some inspiration:
    #    - Check out [streamlit.io](https://streamlit.io)
    #    - Jump into streamlit [documentation](https://docs.streamlit.io)
    #    - Use a neural net to [analyze the Udacity Self-driving Car Image
    #      Dataset] (https://github.com/streamlit/demo-self-driving)
    #    - Explore a [New York City rideshare dataset]
    #      (https://github.com/streamlit/demo-uber-nyc-pickups)
    #    """
    #)
