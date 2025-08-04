import streamlit as st
import face_recognition
import cv2
import numpy as np
from descripteurs import glcm_RGB,haralick_feat_RGB,bitdesc_feat_RGB,concatenation_RGB
from Distances import Recherche_images_similaire
import os

st.markdown("""
            <div>
            <center><h1 style="color:red">Application de recherche d'images similaires</h1></center>
            </div>
            """,unsafe_allow_html=True)

st.markdown("""
            <h4 style="color:blue">S\'il-vous-plait veuillez patienter pendant l'authentification</h4>
            """,unsafe_allow_html=True)
signatures_nom=np.load('SignaturesAll.npy')
signatures=signatures_nom[:,:-1].astype('float')
noms=signatures_nom[:,-1]

menu = st.sidebar.selectbox("Menu",['Authentification','Recherche'])
# auth = False
if "auth" not in st.session_state:
    st.session_state.auth = False

if menu == "Authentification":
    capture=cv2.VideoCapture(0)
    while True:
        name = ''
        reponse,image=capture.read()
        if reponse:
            image_reduit=cv2.resize(image,(0,0),None,0.25,0.25)
            emplacement_face=face_recognition.face_locations(image_reduit)
            cararac_face=face_recognition.face_encodings(image_reduit,emplacement_face)


            # Comparaison de la capture aux signatures
            for encode,loc in zip(cararac_face,emplacement_face):
                match=face_recognition.compare_faces(signatures,encode)
                distance=face_recognition.face_distance(signatures,encode)
                minDist=np.argmin(distance)

                y1,x2,y2,x1=loc
                y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
    
                if match[minDist]==True:
                    cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                    name=noms[minDist]
                    # cv2.putText(image,name,(x1,y2+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    st.session_state.auth = True
                    capture.release()
                    # st.write(f'Bienvenue {name}, vous pouvez a present passer a la recherche')
                    st.markdown(f"""
                    <h5 style="color:green">Bienvenue {name} 😊 vous pouvez a present passer a la recherche</h5>
                    """,unsafe_allow_html=True)
                    break
                else:
                    cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
                    name='Inconnu'
                    # cv2.putText(image,name,(x1,y2+25),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                    st.markdown(f"""
                    <h5 style="color:green">😌☹️Malheureusement nous ne sommes pas en mesure de vous authentifier. Merci d'avoir visité notre site 😊</h5>
                    """,unsafe_allow_html=True)
                    capture.release()
                    break

            
            if st.session_state.auth:
                break
            if name == "Inconnu":
                break
    
elif menu == "Recherche":
    if st.session_state.auth:

        img = st.file_uploader('Televersez une image',type=["jpeg","jpg","png",'bmp'])

        descripteurs = ['GLCM_RGB','Haralick_RGB','BitDescFeat_RGB','Concatenation_RGB']

        distances = ['Euclidienne','Manhattan','Chebychev','Canberra']

        nb_images = None
        descripteur = None
        distance = None

        nb_images = int (st.number_input("Veuillez entrer le nombre d'images similaires a rechercher : "))  
        descripteur = st.selectbox('Veuillez selectionner un descripteur : ', descripteurs)
        distance = st.selectbox('Veuillez selectionner une mesure de distance : ', distances)

        signatures_glcm_rgb = np.load('Signatures_GLCM_RGB.npy')
        signatures_haralick_rgb = np.load('Signatures_haralick_RGB.npy')
        signatures_bitdesc_rgb = np.load('Signatures_bitdesc_RGB.npy')
        signatures_concat_rgb=np.load('Signatures_Concat_RGB.npy')

        resultat = []
        if img is not None and descripteur and distance and nb_images:
            file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
            image_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            carac_glcm_rgb=glcm_RGB(image_cv2)
            carac_haralick_rgb=haralick_feat_RGB(image_cv2)
            carac_bitdesc_rgb=bitdesc_feat_RGB(image_cv2)
            carac_concat_rgb = concatenation_RGB(image_cv2)
            match descripteur:
                case "GLCM_RGB":
                    resultat = Recherche_images_similaire(bdd_signatures=signatures_glcm_rgb, carac_requete=carac_glcm_rgb,Distances=distance,K=nb_images)
                case "Haralick_RGB":
                    resultat = Recherche_images_similaire(bdd_signatures=signatures_haralick_rgb, carac_requete=carac_haralick_rgb,Distances=distance,K=nb_images)

                case "BitDescFeat_RGB":
                    resultat = Recherche_images_similaire(bdd_signatures=signatures_bitdesc_rgb, carac_requete=carac_bitdesc_rgb,Distances=distance,K=nb_images)

                case "Concatenation_RGB":
                    resultat = Recherche_images_similaire(bdd_signatures=signatures_concat_rgb, carac_requete=carac_concat_rgb,Distances=distance,K=nb_images)


        if len(resultat)>0:
            for dist, label, chemin in resultat:
                chemin_absolu = os.path.abspath(f'dataset/{chemin}')
                st.write(f"Distance: {dist:.2f} | Label: {label}")
                st.image(chemin_absolu, caption=f"Label: {label} | Distance: {dist:.2f}", use_column_width=True)
    
    else : 
        st.markdown(f"""
        <h5 style="color:green">☹️ Vous devez d'abord vous authentifier. 😊</h5>
        """,unsafe_allow_html=True)

