# =================================================================
# == INSTITUTO TECNOLOGICO Y DE ESTUDIOS SUPERIORES DE OCCIDENTE ==
# ==         ITESO, UNIVERSIDAD JESUITA DE GUADALAJARA           ==
# ==                                                             ==
# ==            MAESTRÍA EN SISTEMAS COMPUTACIONALES             ==
# ==             PROGRAMACIÓN PARA ANÁLISIS DE DATOS             ==
# ==                 IMPLEMENTACIÓN EN STREAMLIT                 ==
# =================================================================
# ==                 Creacion Dashboard MiBici                   ==
# ==                 Victor Telles | 737066 (AHTyler)            ==
# =================================================================

#----- Importación de Librerías -----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import random
from skimage import io
import os
import chardet

#----- Configuracion inicial---------------------------------------
LOGO_PATH = r'./media/images/MiBici_Logo.png'
DATA_FOLDER = r'data/MiBici-Data'
DATA_NOMENCLATURA_FOLDER = r'data/Nomenclatura-Mibici-Data'
CACHE_FOLDER = r'data/cache'
CACHE_FILE = os.path.join(CACHE_FOLDER, 'datos_procesados.parquet')

# Creacion carpeta si no existe
os.makedirs(CACHE_FOLDER, exist_ok=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Apartado de funciones ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----- Funciones para decodificacion -----------------------------
def dectect_encoding(file):
    '''Funcionalidad para dectectar la codificacion de un archivo'''
    with open(file, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

#----- Funciones para leer y procesamiento de datos --------------
def cargar_datos(data_folder):
    '''Funcionalidad para leer y procesar datos de un archivo CSV'''
    # Formato de las columnas correctas
    columnas_correctas = [
        "Trip_Id", "User_Id", "Gender", "Year_of_Birth",
        "Trip_Start", "Trip_End", "Origin_Id", "Destination_Id"
    ]

    #Diccionario para renombrar columnas
    renombrar_columnas = {
        "Viaje_Id": "Trip_Id",
        "Usuario_Id": "User_Id",
        "Genero": "Gender",
        "Año_de_nacimiento": "Year_of_Birth",
        "A}äe_nacimiento": "Year_of_Birth",
        "AÃ±o_de_nacimiento": "Year_of_Birth",
        "Inicio_del_viaje": "Trip_Start",
        "Fin_del_viaje": "Trip_End",
        "Origen_Id": "Origin_Id",
        "Destino_Id": "Destination_Id"
    }

    #Lista, almacenar todos los dataframe
    dataframes = []

    #Iterar todos los archivos y subcarpetas
    for root, _, files in os.walk(data_folder):
        for file in files:
            # leer archivo CSV
            if file.endswith(".csv"):
                ruta_completa = os.path.join(root, file)
                ruta_completa = ruta_completa.replace("\\", "/")
                #print(f"📂 Intentando leer: {ruta_completa}")

                try:
                    #Dectectar la codificacion del archivo
                    encoding = dectect_encoding(ruta_completa)
                    #print(f"🔍 Codificación detectada: {encoding}")

                    #leer archivo
                    df = pd.read_csv(ruta_completa, encoding=encoding)

                    if df is not None and not df.empty:
                        #print(f"✅ Archivo cargado correctamente: {ruta_completa}")

                        # Renombrar columnas
                        df = df.rename(columns=renombrar_columnas, inplace=False)
                        # Verificar si las columnas están en el orden correcto
                        df = df[[col for col in columnas_correctas if col in df.columns]]

                        #Extraer el año, mes  del archivo
                        nombre_archivo = os.path.basename(ruta_completa)
                        partes = nombre_archivo.split('_')
                        if len(partes) >= 4:
                            anio = int(partes[2])
                            mes = int(partes[3].split('.')[0])
                            df['Year'] = anio
                            df['Month'] = mes

                        #Agregar datos al dataframe
                        dataframes.append(df)
                        st.success(f'✅ Datos cargados: {ruta_completa} - Columnas: {df.columns.tolist()}')
                    else:
                        #print(f"⚠️ Archivo vacío o no se pudo leer: {ruta_completa}")
                        st.error(f'❌ No se pudo leer el archivo: {ruta_completa} esta vacio o no se pudo leer')

                except Exception as e:
                    #print(f"❌ Error leyendo {ruta_completa}: {e}")  # Debugging
                    st.error(f'❌ Error al leer el archivo {ruta_completa}: {e}')

    # Concatenar todos los datos del dataframe en 1
    if dataframes:
        all_data = pd.concat(dataframes, ignore_index=True)
        st.success('Datos cargados y unificados correctamente')
        return all_data
    else:
        st.error('No se pudieron concatenar los datos')
        return  None

#----- Manejamiento Valores nulos --------------------------------
def manejar_valores_nulos(df):
    '''Funcionalidad para manejar valores nulos en el dataframe'''
    if df is not None and not df.empty:
        df = df.dropna(subset=['Trip_Id', 'User_Id', 'Origin_Id', 'Destination_Id'])
    return df

#-----------------------------------------------------------------
#----- Nomenclatura ----------------------------------------------
#-----------------------------------------------------------------
def cargar_nomenclatura(data_nomenclatura_folder):
    '''Funcionalidad para cargar los datos de Nomenclatura de MiBici'''
    nomenclatura_dataframes = []

    # Debug: Verificar si la carpeta existe
    #print(f'📂 Verificando si la carpeta de nomenclatura existe: {os.path.exists(data_nomenclatura_folder)}')
    
    #Cargar y procesar el archivo de nomenclatura
    for root, _, files in os.walk(data_nomenclatura_folder):
        #print(f'📁 Explorando: {root} - Archivos encontrados: {files}')
        for file in files:
            if file.endswith(".csv"):
                #Lectura de archivo
                ruta_completa = os.path.join(root, file)
                ruta_completa = ruta_completa.replace('\\','/')
                #print(f'📂 Intentando leer: {ruta_completa}')

                try:
                    #Dectectar codificador
                    encoding = dectect_encoding(ruta_completa)
                    #print(f'🔍 Codificacion dectectada: {encoding}')

                    #leer CSV
                    df = pd.read_csv(ruta_completa, encoding=encoding)
                    #print(f'✅ Archivo leido correctamente: {ruta_completa}')

                    if df is not None and not df.empty:
                        #añadir datos al dataframe
                        nomenclatura_dataframes.append(df)

                    else:
                        st.error(f'❌ Archivo vacio o no se pudo leer: {ruta_completa}')
                except Exception as e:
                    st.error(f'❌ Error al leer el archivo {ruta_completa}: {e}')
                    #print(f'🛑 Detalles del error: {str(e)}')

    if nomenclatura_dataframes:
        nomenclatura_df = pd.concat(nomenclatura_dataframes, ignore_index=True)
        #print(f' Dataframe final de nomenclatura: {nomenclatura_df.shape}')

        return nomenclatura_df
    else:
        st.error('No se pudieron cargar los datos de Nomenclatura')
        return None

#------------------------------------------------------------------
#----- Agrupacion por estaciones ----------------------------------
#------------------------------------------------------------------

#----- Generar un D.F. para Agrupar las estaciones ----------------
def estaciones():
    '''Funcionalidad para Agrupar (Origin_Id y Destination_Id) con las estaciones'''


#------------------------------------------------------------------
#----- Generar nuevas columnas ------------------------------------
#------------------------------------------------------------------

#----- Funcionalidad para crear Edad ------------------------------

#----- Funcionalidad para crear Tiempo recorrido ------------------

#----- Funcionalidad para calcular distancias ---------------------


#----- Cache  -----------------------------------
@st.cache_data
def save_cache(df, cache_file):
    '''Funcionalidad para guardar el Dataframe en un archivo cache'''
    if df is not None and not df.empty:
        #print(f"💾 Guardando datos en cache: {df.shape}")
        df.to_parquet(cache_file)

@st.cache_data
def load_cache(cache_file):
    '''Funcionalidad para guardar el Dataframe en un archivo cache'''
    if os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            #print(f"✅ Archivo cache cargado correctamente: {df.shape}")
            return df
        except Exception as e:
            #print(f"❌ Error cargando archivo cache: {e}")
            return None
        #return pd.read_parquet(cache_file)
    else:
        #print("⚠️ Archivo cache no encontrado")
        return None


#----- Interfaz App -----------------------------------
def main():
    #----- Configuracion inicial --------------------------------
    st.image(io.imread(LOGO_PATH), width=200)
    st.title('Datos de mi Bici (2014-2024)')
    st.subheader(':blue[MiBici [texto]]')
    st.sidebar.divider()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Apartado Imagen ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----- Configuracion Menu de filtros--------------------------
    st.sidebar.image(io.imread(LOGO_PATH), width=200)
    st.sidebar.markdown('## MENU DE FILTROS')
    st.sidebar.divider()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Apartado Cache ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----- Apartado de cache -------------------------------------
    # =========== SIDEBAR ========================================
    st.sidebar.markdown('### Cache')
    opcion_cache = st.sidebar.radio(
        'Seleccione una opcion:',
        ["Crear nuevos datos", "Cargar datos"],
        index=1
    )
    st.sidebar.divider()
    # =========== FIN SIDEBAR ====================================

    #----- Cargar o Procesar datos -------------------------------
    df = None
    if opcion_cache == "Crear nuevos datos":
        with st.spinner('Procesando datos desde la carpeta de origen...'):
            df = cargar_datos(DATA_FOLDER)
            df = manejar_valores_nulos(df)
            if df is not None and not df.empty:
                save_cache(df, CACHE_FILE)
                st.success('Datos procesados y guardados en cache')
            else:
                st.sidebar.write('Cargando datos desde el cache...')
    else:
        with st.spinner('Cargando datos desde el cache...'):
            df = load_cache(CACHE_FILE)
            if df is not None and not df.empty:
                st.success('Datos cargados desde el cache')
                #print(f"📊 Datos cargados correctamente desde cache: {df.shape}")
            else:
                st.error('No se encontraron datos en el cache. Seleccione \'Crear nuevos datos\'.')
                #print("⚠️ df está vacío o None después de intentar cargar el cache")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Apartado Filtro ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----- Filtrado de opcion ------------------------------------
    # =========== SIDEBAR ========================================
    st.sidebar.markdown('### Filtrado')
    opcion_filtrado = st.sidebar.radio(
        'Seleccione una opcion de filtrado.',
        ['Año x Meses', 'Mes x Años']
    )
    st.sidebar.divider()
    #Aplicar filtros
    if df is not None and not df.empty:
        #----- Filtro de Año x Meses -----------------------------
        if opcion_filtrado == 'Año x Meses':
            # Año
            year_avaliable = df['Year'].unique()
            year_selected = st.sidebar.selectbox('Selecciona el año:', year_avaliable)
            #Meses
            month_avaliable = df[df['Year'] == year_selected]['Month'].unique()
            month_selected = st.sidebar.multiselect('Selecciona los meses:', month_avaliable, default=month_avaliable)
            #Aplicar filtro
            datos_filtrados = df[(df['Year'] == year_selected) & (df['Month'].isin(month_selected))]

        else:
        #----- Filtro de Mes x Años ------------------------------
            #Mes
            month_avaliable = df['Month'].unique()
            month_selected = st.sidebar.selectbox('Selecciona el mes', month_avaliable)
            #Años
            year_avaliable = df[df['Month'] == month_selected]['Year'].unique()
            year_selected = st.sidebar.multiselect('Selecciona los años', year_avaliable, default=year_avaliable)
            #Aplicacion filtro
            datos_filtrados = df[(df['Month'] == month_selected) & (df['Year'].isin(year_selected))]
    # =========== FIN SIDEBAR ====================================

        #Mostrando y aplicando filtros
        st.write(f'Año: {year_selected}')
        st.write(f'Meses: {month_selected}')
        st.write(datos_filtrados)

    #----- Cargar Nomenclatura  ---------------------------------
    with st.spinner('Cargando datos de nomenclatura...'):
        nomenclatura_df = cargar_nomenclatura(DATA_NOMENCLATURA_FOLDER)
        if nomenclatura_df is not None and not nomenclatura_df.empty:
            save_cache(nomenclatura_df, os.path.join(CACHE_FOLDER, 'nomenclatura_cache.parquet'))
            st.success('Nomenclatura cargada y guardada en cache')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Apartado de Nomenclatura ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----- Mostrar datos de nomenclatura -------------------------
    # =========== SIDEBAR ========================================
    st.sidebar.divider()
    st.sidebar.markdown('### Mostrar Nomenclatura')
    mostrar_nomenclatura = st.sidebar.toggle('Mostrar Datos de Nomenclatura MiBici', value=False)
    # =========== FIN SIDEBAR ====================================


    # =========== CONTENIDO ======================================
    if mostrar_nomenclatura and nomenclatura_df is not None:
        st.write('### Datos de Nomenclatura')
        st.dataframe(nomenclatura_df)
    # =========== FIN CONTENIDO ==================================


#----- Ejecución de la Aplicación --------------------------------
if __name__ == '__main__':
    main()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Apartado de Graficos ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#=================================================================
#===== Grafica Lineal ==== Numero de viajes * Mes y año ==========
#=================================================================

#=================================================================
#===== Grafica Barras ==== Promedio de viaje (mes/dia * semana) ==
#=================================================================

#=================================================================
#===== Grafica Histograma ==== Distancia recorrida ===============
#=================================================================

#=================================================================
#===== Grafica Histograma ==== Hombres vs Mujeres = Uso de MiBici=
#=================================================================

#=================================================================
#===== Grafica Boxplot ==== Tiempo de viaje vs Ruta y genero =====
#=================================================================

#=================================================================
#===== Grafica Barras ==== Uso por Dias de la semana =============
#=================================================================

#=================================================================
#===== Grafico Correlacion ==== Uso de estanciones (Inicio / Fin)=
#=================================================================

#=================================================================
#===== Grafico Correlacion ==== Correlacion Dia de la semanas ====
#=================================================================

