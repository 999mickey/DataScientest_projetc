import streamlit as st
from fire_state import create_store, form_update , set_state , get_state

slot = "home_page"
def display_user_selection(filter):
    with st.form("Nombre "):
        st.slider("Nombre d utilisateur à présenter",min_value= 1,max_value= 10, value=5, step=1, key="num_user_to_present")                        

        st.form_submit_button(label="Valider", on_click=form_update, args=(slot,))

    nuser = st.session_state["num_user_to_present"]
    print("nombre d'utlisateur = ",st.session_state["num_user_to_present"])

    userIds =[]
    userIds = filter.get_best_voter_users(nuser)
    st.dataframe(userIds)        

    userIds = userIds[filter.user_key_name]
    userIds = userIds.to_list()
    userId = ''
    userIds.insert(0, userId)
    
    userId = st.selectbox('Choix de l utilisateur', userIds,key="user_id")
    print("Utilisateur selectionné => ",userId)        
    return userId

def display_nb_pres(maxp=None):
    maxval = 10
    if maxp != None:
        maxval = maxval
    with  st.form("Nombre 3"):                
        st.slider("Nombre à présenter",min_value= 1,max_value= maxval,value=5, step=1, key="nb")
        st.form_submit_button(label="Valider", on_click=form_update, args=(slot,))


def display_nb_pres_selection():
    with st.form("Nombre 2"):                
        st.slider("Nombre d utilisateur similaires",min_value= 1,max_value= 10,value=5, step=1, key="nb_similar_user")
        
        st.slider("Nombre de préférences", min_value= 1,max_value= 10,value=5, step=1, key="nb_pref_user")

        st.slider("Nombre de prédiction user", min_value= 1,max_value= 10,value=5, step=1, key="nb_pred_user")
        
        st.slider("Nombre de prédiction Item", min_value= 1,max_value= 10,value=5, step=1, key="nb_pred_item")
                
        st.form_submit_button(label="Valider", on_click=form_update, args=(slot,))

    k_close_param = st.session_state["nb_similar_user"]
    npref = st.session_state["nb_pref_user"]
    npreditem = st.session_state["nb_pred_item"]
    npreduser = st.session_state["nb_pred_user"]
    return k_close_param ,npref , npreditem , npreduser

def display_predictors():
    from surprise import NormalPredictor
    from surprise import SVD
    with st.form("predictor"):                
        predictors = ["","NormalPredictor()","SVD()"]
        predictor = st.selectbox('Choix de predictor', predictors,key="predictor")
        st.form_submit_button(label="Valider", on_click=form_update, args=(slot,))
        ltxt = "Le predicteur choisi est :"+predictor
        st.text(ltxt)
    if predictor != '':
        return  eval(predictor) ,predictor
    return None , None

#def display_trainig_params(predicor_name):
#    if predicor_name == "SVD()":





