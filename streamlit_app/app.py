from collections import OrderedDict

import streamlit as st

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config

style_path ="C:/Users/engizulu/Documents/Projet_Datascientest/structural-protein-classification/streamlit_app/"

st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
    layout="centered",  # wide,
    initial_sidebar_state="auto"
)

# TODO : you can (and should) rename and add tabs in the ./tabs folder, and import them here.
from tabs import intro, tab_eda, tab_model_ml, tab_model_deep, tab_demo_ml, tab_demo_deep


with open(style_path + 'style.css', "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (tab_eda.sidebar_name, tab_eda),
        (tab_model_ml.sidebar_name, tab_model_ml),
        (tab_demo_ml.sidebar_name, tab_demo_ml),
        (tab_model_deep.sidebar_name, tab_model_deep),
        (tab_demo_deep.sidebar_name, tab_demo_deep)
    ]
)


def run():
    st.sidebar.image(
        "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
        #width=200,
        use_column_width=True
    )
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Auteurs:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
