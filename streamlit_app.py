from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Cargar el archivo CSV desde una ruta específica
file_path = 'BBDD TA - BD.csv'
df = pd.read_csv(file_path, sep=',', header=0, index_col=0)

# Seleccionar columnas de tipo 'object'
object_columns = df.select_dtypes(include=['object']).columns

# Filtrar columnas de tipo 'object' que tienen valores faltantes
cf = [col for col in object_columns if df[col].isnull().any()]

# Imputar los valores faltantes con la moda
from sklearn.impute import SimpleImputer
imputador = SimpleImputer(strategy='most_frequent')
df[cf] = imputador.fit_transform(df[cf])

# Definir la función para transformar los datos
def transformar_datos(df):
    # Reemplazar caracteres específicos
    df = df.replace(',', '.', regex=True)
    df = df.replace('-', '/', regex=True)

    # Filtrar datos
    df = df[df['Rol'] == 'Comercial']
    df = df[df['PAIS'] == 'MEXICO']
    df = df[df['EMPRESA'] == 'SELLERS']

    # Guardar variables originales antes de la transformación
    df_original = df.copy()

    # Conversión de tipos de datos
    df['SALARIO_BRUTO'] = df['SALARIO_BRUTO'].astype('float64')
    df['Cantidad de Transacciones'] = df['Cantidad de Transacciones'].astype('float64')
    df['Meta'] = df['Meta'].astype('float64')
    df['NIVEL'] = df['NIVEL'].astype('object')
    df['FECHA DE INGRESO'] = pd.to_datetime(df['FECHA DE INGRESO'], format='%d/%m/%Y')
    df['FECHA DE RETIRO'] = pd.to_datetime(df['FECHA DE RETIRO'], format='%d/%m/%Y')
    fecha_hoy = pd.Timestamp('today')
    df['FECHA DE RETIRO'].fillna(fecha_hoy, inplace=True)

    # Cálculo de nuevas características
    df['CVR'] = df['Cantidad de Transacciones'] / df['Meta']
    df['CVR'] = df['CVR'].fillna(0)
    df['Salario_USD'] = np.where(df['PAIS'] == 'COLOMBIA', df['SALARIO_REFERENTE'] / 4000, np.nan)
    df['diferencia_dias'] = (df['FECHA DE RETIRO'] - df['FECHA DE INGRESO']).dt.days
    # Asegurando que 'diferencia_dias' sea de tipo float64
    df['diferencia_dias'] = df['diferencia_dias'].astype('float64')

    return df, df_original

# Aplicar la función de transformación
df_transformed, df_original = transformar_datos(df)

def preparar_datos_modelo(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    scaler = StandardScaler()
    df['CVR_estandarizada'] = scaler.fit_transform(df[['CVR']])

    # Aplicar KBinsDiscretizer a la variable "CVR" estandarizada
    kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')

    # Handle NaN values before applying KBinsDiscretizer
    df['CVR_estandarizada'] = df['CVR_estandarizada'].fillna(df['CVR_estandarizada'].mean()) # Replace NaN with mean

    df['CVR_binned'] = kbd.fit_transform(df[['CVR_estandarizada']])

    # Clustering
    # Reshape de la columna para el algoritmo de clustering
    X = df['CVR'].values.reshape(-1, 1)

    # Imputar valores faltantes (NaN) con la media
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Aplicar K-means con, por ejemplo, 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X_imputed)
    df['CVR_cluster'] = kmeans.labels_
    columnas_numericas = df.select_dtypes(include=['float64', 'int64'])

    # Agregar la columna 'categoria' al DataFrame de columnas numéricas
    columnas_numericas['CVR_binned'] = df['CVR_binned']
    # Check if 'ESCOLARIDAD' column exists before processing
    if 'ESCOLARIDAD' in df.columns:
        # Codificación de 'ESCOLARIDAD'
        orden_escolaridad = {
            'PRIMARIA': 0,
            'BACHILLER': 1,
            'TECNICO': 2,
            'TECNÓLOGO': 3,
            'PREGRADO': 4,
            'POSTGRADO': 5
        }
        # Map 'ESCOLARIDAD' to 'ESCOLARIDAD_Numerica' before dropping 'ESCOLARIDAD'
        df['ESCOLARIDAD_Numerica'] = df['ESCOLARIDAD'].map(orden_escolaridad)
        df.drop('ESCOLARIDAD', axis=1, inplace=True) # Now you can safely drop 'ESCOLARIDAD'

    # Selección de columnas relevantes
    columns_to_keep = [
        'HIJOS', 'GENERO', 'Fuente de Reclutamiento', 'Tipo de Contacto',
        'ESCOLARIDAD_Numerica', 'EDAD', 'CVR_cluster','CVR_binned','SECTOR','LIDER'
    ]
    df = df[columns_to_keep]

    # Imputación de valores faltantes
    df_float = df.select_dtypes(include=['float64', 'int32'])
    imputador_knn = KNNImputer(n_neighbors=5)
    data_imp = pd.DataFrame(imputador_knn.fit_transform(df_float), columns=df_float.columns, index=df_float.index)

    # Si no hay columnas numéricas, simplemente se copia el DataFrame original
    if df_float.empty:
        data_imp = df_float

    # Codificación de variables categóricas a dummies
    df_object = df.select_dtypes(include=['object'])
    datos_dummies = pd.get_dummies(df_object, columns=['HIJOS', 'GENERO', 'Fuente de Reclutamiento', 'Tipo de Contacto','SECTOR','LIDER'])

    # Combinar las variables imputadas y las dummies
    df_com = pd.concat([data_imp, datos_dummies], axis=1)

    # Separar características (X) y la variable objetivo (y)
    X = df_com.drop(columns=['CVR_binned'])
    y = df_com['CVR_binned']

    # Aplicación de SMOTE para sobremuestreo si hay suficientes datos y clases

    smote = SMOTE(sampling_strategy='not majority', random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Crear un nuevo DataFrame con los datos balanceados
    df_res = pd.DataFrame(X_res, columns=X.columns)
    df_res['CVR_binned'] = y_res


    # Filtrado de variables con baja correlación con 'CVR_cluster'
    correlation_matrix = df_res.corr()
    correlation_threshold = 0.5
    low_correlation_vars = correlation_matrix[abs(correlation_matrix['CVR_binned']) < correlation_threshold]['CVR_binned']
    low_correlation_var_names = low_correlation_vars.index.tolist()

    # Asegurar que 'CVR_cluster' esté en la lista de variables seleccionadas
    if 'CVR_binned' not in low_correlation_var_names:
        low_correlation_var_names.append('CVR_binned')

    # Crear un nuevo DataFrame con las variables seleccionadas
    df_low_corr = df_res[low_correlation_var_names]

    # Eliminar columnas específicas si están presentes
    columns_to_drop = ['Meta', 'diferencia_dias']
    df_low_corr.drop(columns=[col for col in columns_to_drop if col in df_low_corr.columns], axis=1, inplace=True)

    # Definir las variables predictoras (X) y la variable objetivo (y)
    X = df_low_corr.drop(columns=['CVR_binned'])
    y = df_low_corr['CVR_binned']

    return X, y


##Modelo

# Definir el modelo Random Forest
def entrenar_modelo(X, y):
    param_grid = {
        'n_estimators': [200],
        'max_depth': [20],
        'min_samples_split': [10],
        'min_samples_leaf': [1]
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    random_forest_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_random_forest_model = RandomForestClassifier(**best_params, random_state=42)
    best_random_forest_model.fit(X_train, y_train)

    return best_random_forest_model, X_test, y_test

# Cargar los datos y entrenar el modelo
df = transformar_datos(df)[0]
X, y = preparar_datos_modelo(df)
modelo, X_test, y_test = entrenar_modelo(X, y)

# Aplicación Streamlit
st.title("Predicción de Calidad de Nuevos Ingresos")

# Formulario para ingresar nuevos datos
st.sidebar.header("Ingrese los datos del nuevo ingreso")
HIJOS = st.sidebar.selectbox("HIJOS", ['NO', 'SI'], key='selectbox_hijos')
GENERO = st.sidebar.selectbox("GENERO", ['MASCULINO', 'FEMENINO'], key='selectbox_genero')
Fuente_de_Reclutamiento = st.sidebar.selectbox("Fuente de Reclutamiento", df['Fuente de Reclutamiento'].unique(), key='selectbox_fuente')
Tipo_de_Contacto = st.sidebar.selectbox("Tipo de Contacto", df['Tipo de Contacto'].unique(), key='selectbox_contacto')
SECTOR= st.sidebar.selectbox("SECTOR", df['SECTOR'].unique(), key='selectbox_sector')
LIDER = st.sidebar.selectbox("LIDER", df['LIDER'].unique(), key='selectbox_lider')
edad = st.sidebar.slider("Edad", 18, 70, key='slider_edad')
# Mapeo de los niveles de escolaridad a números
escolaridad_map = {
    'PRIMARIA': 0,
    'BACHILLER': 1,
    'TECNICO': 2,
    'TECNÓLOGO': 3,
    'PREGRADO': 4,
    'POSTGRADO': 5
}

# Selectbox para Escolaridad usando el mapeo
escolaridad_seleccionada = st.sidebar.selectbox("Escolaridad", list(escolaridad_map.keys()), key='selectbox_escolaridad')
escolaridad_numrica = escolaridad_map[escolaridad_seleccionada]
TIEMPO_ESTUDIO = st.sidebar.slider("TIEMPO ESTUDIO", 0, 20, key='slider_tiempo_estudio')
EXPERIENCIA = st.sidebar.slider("EXPERIENCIA", 0, 20, key='slider_experiencia')

# Convertir las variables ingresadas en un DataFrame
nuevo_dato = pd.DataFrame({
    'HIJOS': [HIJOS],
    'GENERO': [GENERO],
    'Fuente de Reclutamiento': [Fuente_de_Reclutamiento],
    'Tipo de Contacto': [Tipo_de_Contacto],
    'SECTOR': [SECTOR],
    'LIDER': [LIDER],
    'TIEMPO ESTUDIO': [TIEMPO_ESTUDIO],
    'EXPERIENCIA': [EXPERIENCIA],
    'edad': [edad],
    'escolaridad_numrica': [escolaridad_numrica]
})

# Procesar el nuevo dato
nuevo_dato = pd.get_dummies(nuevo_dato)
missing_cols = set(X.columns) - set(nuevo_dato.columns)
for col in missing_cols:
    nuevo_dato[col] = 0
nuevo_dato = nuevo_dato[X.columns]

# Cargar los datos y entrenar el modelo
df = transformar_datos(df)[0]
X, y = preparar_datos_modelo(df)
modelo, X_test, y_test = entrenar_modelo(X, y)

# Diccionario para mapear los nombres de las clases
nombres_categorias = {
    0: "Alto CVR",
    1: "Bajo CVR",
    2: "Medio CVR"
}

# Realizar la predicción
if st.sidebar.button("Realizar Predicción"):
    # Obtener las probabilidades predichas para cada clase
    probabilidades = modelo.predict_proba(nuevo_dato)
    
    # Crear un DataFrame para mostrar las probabilidades de cada clase
    probabilidades_df = pd.DataFrame(probabilidades, columns=[nombres_categorias[clase] for clase in modelo.classes_])
    st.write("Probabilidades de cada categoría de CVR:")
    st.write(probabilidades_df)
    
    # Mostrar la categoría con la mayor probabilidad
    prediccion = modelo.predict(nuevo_dato)
    categoria_predicha = nombres_categorias[prediccion[0]]
    st.write(f"La categoría de CVR estimada es: {categoria_predicha}")
