
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

#----- Funciones para decode -----------------------------------
def dectect_encoding(file):
    '''Funcionalidad para dectectar la codificacion de un archivo'''
    with open(file, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

#----- Funciones para leer y procesaro datos -----------------------------------
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

    #Lista, almacentar todos los dataframe
    dataframes = []

    #Iterar todos los archivos y subcarpetas
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            # Lectura de archivo CSV
            if file.endswith(".csv"):
                ruta_completa = os.path.join(root, file)
                try:
                    #Dectectar la codificacion del archivo
                    encoding = dectect_encoding(ruta_completa)
                    st.write(f'Dectectada codificacion para {ruta_completa}: {encoding}')

                    #leer archivo
                    df = pd.read_csv(ruta_completa, encoding=encoding)
                    st.write(f'Archivo leido: {ruta_completa}-Fila: {len(df)}')

                    if df is None or df.empty:
                        st.error(f'❌ El archivo {ruta_completa} esta vacio o no se pudo leer correctamente')
                        continue

                    # Renombrar columnas
                    df = df.rename(columns=renombrar_columnas, inplace=False)
                    st.write(f'Columnas renombradas: {df.columns.tolist()}')

                    # Verificar si las columnas están en el orden correcto
                    df = df[[col for col in columnas_correctas if col in df.columns]]
                    st.write(f'Columnas seleccionadas: {df.columns.tolist()}')

                    #Extraer el año, mes  del archivo
                    nombre_archivo = os.path.basename(ruta_completa)
                    partes = nombre_archivo.split('_')
                    if len(partes) >= 4:
                        anio = int(partes[2])
                        mes = int(partes[3].split('.')[0])
                        df['Year'] = anio
                        df['Month'] = mes
                        st.write(f' Año y mes extraidos: Año = {anio}, Mes = {mes}')

                        #Agregar datos al dataframe
                        dataframes.append(df)
                        st.success(f'✅ Datos cargados: {ruta_completa} - Columnas: {df.columns.tolist()}')
                    else:
                        st.error(f'❌ No se pudo leer el archivo: {ruta_completa} esta vacio o no se pudo leer')

                except Exception as e:
                    st.error(f'❌ Error al leer el archivo {ruta_completa}: {e}')

    # Concatenar todos los datos del dataframe en 1
    if dataframes:
        all_data = pd.concat(dataframes, ignore_index=True)
        st.success('Datos cargados y unificados correctamente')
        return all_data
    else:
        st.error('No se pudieron concatenar los datos')
        return  None



#----- Lectura de datos -----------------------------------
#Ruta de los datos de MiBici
data_folder = r'data\MiBici-Data'
#data_folder = r'data\MiBici-Data\2014\datos_abiertos_2014_12.csv'
#print(f'Ruta de la carpeta: {data_folder}')

#cargar losd atos
df = cargar_datos(data_folder)


#Mostrar los datos cargados
if df is not None:
    st.write('Datos cargados:')
    st.write(df.head())


#------------------------------------------------------------------
#----- Configuración Sidebar --------------------------------------
#------------------------------------------------------------------


#------------------------------------------------------------------
#----- Configuración Inicial del Panel Central --------------------
#------------------------------------------------------------------

#----- Lectura de la Imagen ---------------------------------------
Logo = io.imread(r"./media/images/MiBici_Logo.png")

#----- Renderizado de la Imagen -----------------------------------
st.image(Logo, width = 200)

#----- Renderizado del Texto --------------------------------------
st.title("Uso Básico de Streamlit")
st.subheader(":blue[Streamlit es un *framework* para la creación de aplicaciones web "
             "interactivas y basadas en datos.]")


#------------------------------------------------------------------
#----- Configuración de los Elementos del DashBoard ---------------
#------------------------------------------------------------------

#----- Renderizado de la Imagen y el Título en el Dashboard -------
st.sidebar.image(Logo, width = 200)
st.sidebar.markdown("## MENU DE FILTROS")
st.sidebar.divider()

#----- Selector del Mes -------------------------------------------
vars_mes = ['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC']
default_hist = vars_mes.index('ENE')
histo_selected = st.sidebar.selectbox('Elección del Mes para el Histograma:', vars_mes, index = default_hist)
st.sidebar.divider()

#----- GRÁFICO DE LÍNEAS ------------------------------------------
#----- Selector de las Personas -----------------------------------


#----- GRÁFICO DE CORRELACIÓN -------------------------------------
#----- Selector del Mapa de Color ---------------------------------


#----- Selector de los Meses para el Histograma -------------------
mes_multi_selected = st.sidebar.multiselect('Elementos de la Matriz de Correlación:', vars_mes, default = vars_mes)


#------------------------------------------------------------------
#----- Configuración de Texto y Elementos del Panel Central -------
#------------------------------------------------------------------

#----- Lectura de los Datos Desde el Archivo CSV ------------------

#----- Renderizado del Texto --------------------------------------

#----- Renderizado del DataFrame ----------------------------------

#------------------------------------------------------------------
#----- Configuración de los Elementos del Panel Central -----------
#------------------------------------------------------------------

#----- Apartado de diferentes Graficos  ------------------------------------------------
#----- HISTOGRAMA POR MES -----------------------------------------

#----- GRÁFICO DE LÍNEAS -----------------------

#----- GRÁFICO DE CORRELACIÓN DE LOS MESES ------------------------

#Generación del gráfico

#Renderización del gráfico
