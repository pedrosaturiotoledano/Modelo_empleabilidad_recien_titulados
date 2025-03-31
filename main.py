import streamlit as st
import xgboost as xgb
import numpy as np

# Ejecutar en la terminal el siguiente comando - streamlit run main.py

# Configuración de la página
st.set_page_config(page_title="Modelo de predicción de Empleabilidad", page_icon="🎓", layout="centered")

# Cargar el modelo XGBoost
@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model("model.xgb")  
    return model

model = load_model()

# Título
st.title("🎓 Empleabilidad en estudiantes recien titulados")
st.text ('En este proyecto, se utilizó el modelo XGBoost para predecir si un alumno, al terminar sus estudios, será contratado, basándose en diversos factores como el número de prácticas,'
' proyectos, certificaciones, calificaciones, entre otros. El modelo fue entrenado con un dataset de Kaggle que contiene 10000 muestras. Con un accuracy de 0.804, el modelo demostró el mejor'
' rendimiento en términos de precisión. Su función es clasificar a los alumnos en dos categorías: aquellos con altas probabilidades de ser contratados y aquellos con bajas probabilidades, '
'proporcionando así una herramienta útil para prever el éxito laboral de los estudiantes al finalizar sus estudios.')
st.markdown("#### 📊 Introduce sus datos para obtener la probabilidad de ser contratado al terminar los estudios:")

# Lista de características del modelo
features = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 
            'AptitudeTestScore', 'SoftSkillsRating', 'ExtracurricularActivities', 
            'PlacementTraining', 'SSC_Marks', 'HSC_Marks']

# Entrada de datos
user_input = []
for feature in features:
    if feature == "CGPA":
        value = st.number_input(f"Nota media (0-10)", min_value=0.00, max_value=10.00, value=0.00, step=0.01, format="%.2f")
    elif feature == "Internships":
        value = st.selectbox(f"Número de prácticas", [0, 1, 2])  
    elif feature == "Projects":
        value = st.selectbox(f"Número de proyectos", [0, 1, 2, 3])  
    elif feature == "Workshops/Certifications":
        value = st.selectbox(f"Número de certificaciones", [0, 1, 2, 3, 4, 5, 6, 7])  
    elif feature in ["AptitudeTestScore"]:
        value = st.number_input(f"Puntuación en el Test de Aptitud (0-100)", min_value=0, max_value=100, value=0, step=1)
    elif feature in ["SSC_Marks"]:
        value = st.number_input(f"Nota media secundaria (0-100)", min_value=0, max_value=100, value=0, step=1)  
    elif feature in ["HSC_Marks"]:
        value = st.number_input(f"Nota media bachillerato (0-100)", min_value=0, max_value=100, value=0, step=1)    
    elif feature == "SoftSkillsRating":
        value = st.number_input(f"Puntuación en soft-skills (0-5)", min_value=0.0, max_value=5.0, value=0.0, step=0.1, format="%.1f")  
    elif feature in ["ExtracurricularActivities"]:
        value = st.selectbox(f"Actividades extraescolares", ["Sí", "No"])  
        value = True if value == "Sí" else False
    elif feature in ["PlacementTraining"]:
        value = st.selectbox(f"Formación para búsqueda de empleo", ["Sí", "No"])  
        value = True if value == "Sí" else False    
    user_input.append(value)

# Botón de predicción
st.markdown("---")
if st.button("🚀 Obtener Predicción", use_container_width=True):
    data = xgb.DMatrix(np.array([user_input], dtype=np.float32))  
    prediction = model.predict(data)
    
    # Mostrar resultado con estilo (sin color rojo)
    st.markdown(f"""
        <div style="text-align:center; padding:20px; background-color:#eef4ff; border-radius:10px;">
            <h2 style="color:#007bff;">🔍 Probabilidad de Empleabilidad</h2>
            <h1 style="color:#008000;">{prediction[0]:.2%}</h1>
        </div>
    """, unsafe_allow_html=True)
