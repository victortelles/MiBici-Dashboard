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
LOGO_PATH_AGE = r'./media/images/Edad.png'
IMAGE_PATH_STATION = r'./media/images/Estacion.png'
DATA_FOLDER = r'data/MiBici-Data/Test'
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
        "Aï¿½o_de_nacimiento": "Year_of_Birth",
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
                        #print(f"📊 Columnas renombradas y ordenadas correctamente {df.columns}")

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Nomenclatura ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Manejamiento de valores vacios/null ~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def manejar_valores_nulos_mibici(df):
    '''Funcionalidad para manejar valores nulos en el dataframe de "MiBici" '''
    if df is not None and not df.empty:
        #no pueden contener valores nulos
        columnas_criticas = ["Trip_Id", "User_Id", "Origin_Id", "Destination_Id"]

        #Eliminar filas con valor nulos
        df = df.dropna(subset=columnas_criticas)

        #Rellenar valores nulos
        df = df.ffill()

        st.success('Valores Nulos en datos de MiBici Manejados correctamente')
    return df

def manejar_valors_nulos_nomenclatura(df):
    '''Funcionalidad para manejar valores nulos en el dataframe de "Nomenclatura"'''
    if df is not None and not df.empty:
        #no pueden contener valores nulos
        columnas_criticas = ["id", "name", "latitude", "longitude"]

        # Eliminar filas con valor nulos
        df = df.dropna(subset=columnas_criticas)

        #Rellenar valores nulos
        df = df.ffill()

        st.success('Valores Nulos en datos de Nomenclatura Manejados correctamente')
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Manejamiento incosistencias ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def manejar_fecha(df):
    '''Funcionalidad para manejar inconsistencia de fecha y hora inicio/fin '''
    if df is not None and not df.empty:
        # Convertir las columnas de fecha a datetime
        df['Trip_Start'] = pd.to_datetime(df['Trip_Start'], format='mixed', errors='coerce')
        df['Trip_End'] = pd.to_datetime(df['Trip_End'], format='mixed', errors='coerce')

        # Eliminar filas con fechas inválidas
        df = df.dropna(subset=['Trip_Start', 'Trip_End'])

        # Eliminar datos de prueba (horas perfectas (12 y 15))
        df = df[~(
            (df['Trip_Start'].dt.strftime('%H:%M:%S') == '12:00:00') & 
            (df['Trip_End'].dt.strftime('%H:%M:%S') == '15:00:00')
        )]

        # Asegurar el formato AAAA-MM-DD HH:MM:SS
        df['Trip_Start'] = df['Trip_Start'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['Trip_End'] = df['Trip_End'].dt.strftime('%Y-%m-%d %H:%M:%S')

        #print('Inconsistencias en fechas corregidas correctamente')
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Agrupacion por estaciones ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----- Generar un D.F. para Agrupar las estaciones ---------------
def estaciones(df_mibici, df_nomenclatura):
    '''Funcionalidad para Agrupar (Origin_Id y Destination_Id) con las estaciones (nomenclatura (id))'''
    # Seleccionar columnas relevantes de nomenclatura
    df_nomenclatura = df_nomenclatura[['id', 'name']].rename(columns={'id': 'Station_Id', 'name': 'Station_Name'})

    # Unir datos de MiBici con nomenclatura para obtener el nombre de la estación de origen
    df_mibici = df_mibici.merge(df_nomenclatura, left_on='Origin_Id', right_on='Station_Id', how='left') \
                        .rename(columns={'Station_Name': 'Origin_Station'}) \
                        .drop(columns=['Station_Id'])

    # Unir datos de MiBici con nomenclatura para obtener el nombre de la estación de destino
    df_mibici = df_mibici.merge(df_nomenclatura, left_on='Destination_Id', right_on='Station_Id', how='left') \
                        .rename(columns={'Station_Name': 'Destination_Station'}) \
                        .drop(columns=['Station_Id'])

    return df_mibici[['Trip_Id', 'Origin_Id', 'Origin_Station', 'Destination_Id', 'Destination_Station']]

#----- Generar un conteo x Estacion ------------------------------
def conteo_estacion(df_mibici, df_nomenclatura, tipo):
    '''Funcionalidad para generar un conteo de viajes por estaciones'''
    # Cargar los datos
    df_agrupado = estaciones(df_mibici, df_nomenclatura)

    # Si el df esta vacio, regresar None
    if df_agrupado is None or df_agrupado.empty:
        return None

    if tipo == 'Salen':
        # Contar la cantidad de viajes por estación de salida
        conteo = df_agrupado['Origin_Id'].value_counts().reset_index()
        conteo.columns = ['Origin_Id', 'OutCount_Station']

        # Juntar el Id y contador con los nombres de las estaciones de origen
        conteo = conteo.merge(df_agrupado[['Origin_Id', 'Origin_Station']].drop_duplicates(), on='Origin_Id', how='left')
        return conteo[['Origin_Id', 'Origin_Station', 'OutCount_Station']]

    elif tipo == 'Llegan':
        # Contar la cantidad de viajes por estación de llegada
        conteo = df_agrupado['Destination_Id'].value_counts().reset_index()
        conteo.columns = ['Destination_Id', 'InCount_Station']

        # Juntar el Id y contador con los nombres de las estaciones de destino
        conteo = conteo.merge(df_agrupado[['Destination_Id', 'Destination_Station']].drop_duplicates(), on='Destination_Id', how='left')
        return conteo[['Destination_Id', 'Destination_Station', 'InCount_Station']]

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Generar nuevas columnas ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----- Funcionalidad para crear columna Edad ----------------------
def edad(df):
    '''Funcionalidad para añadir columna de la edad del usuario'''
    if 'Year_of_Birth' in df.columns:
        #df['Age'] = pd.to_numeric(df['Year_of_Birth'], erros='coerce') #Convertir numerico
        df['Age'] = 2025 - df['Year_of_Birth']
        df.loc[(df['Age'] < 0) | (df['Age']> 150), 'Age'] = pd.NA # Filtrar valores erroneos (Limite excedido)
        return df
    else:
        st.error('❌ No se pudo calcular la edad porque la columna "Year_of_Birth" no esta disponible')
        return df

#----- Funcionalidad para crear columna de Tiempo recorrido --------
def tiempo_recorrido(df):
    '''Funcionalidad para añadir columna de tiempo recorrido ()'''

#----- Funcionalidad para calcular distancias ---------------------
def distancia():
    '''Funcionalidad para añadir una columna para saber la distancia ()'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Apartado de cache ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


#~~~~~~~~~~~~~~~~~~~~~~~~~ Interfaz APP ~~~~~~~~~~~~~~~~~~~~~~~~~
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
    # =========== SIDEBAR ========================================
    st.sidebar.image(io.imread(LOGO_PATH), width=200)
    st.sidebar.markdown('## MENU DE FILTROS')
    st.sidebar.divider()
    # =========== FIN SIDEBAR ====================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Apartado Cache ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            df = manejar_fecha(df)
            df = manejar_valores_nulos_mibici(df)
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
            nomenclatura_df = manejar_valors_nulos_nomenclatura(nomenclatura_df)
            save_cache(nomenclatura_df, os.path.join(CACHE_FOLDER, 'nomenclatura_cache.parquet'))
            st.success('Nomenclatura limpiada, cargada y guardada en cache')

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Apartado Estaciones ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----- Mostrar datos de agrupaciones de estaciones -----------
    # =========== CONTENIDO ======================================
    st.divider()
    st.markdown('### Datos de estaciones')
    st.image(io.imread(IMAGE_PATH_STATION), width=800)
    st.text('Visualización de las estaciones con sus respectivas agrupaciones.')

    #Validar si hay datos en los df.
    if df is not None and nomenclatura_df is not None:
        estaciones_df = estaciones(df, nomenclatura_df)
        if estaciones_df is not None and not estaciones_df.empty:
            st.dataframe(estaciones_df)
        else:
            st.warning('No se encontraron datos de estaciones')
    else:
        st.error('❌ No se pudo calcular estaciones debido a datos faltantes.')

    #~~~~~ Contador de viajes por estación ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.divider()
    st.markdown('### Contador de estaciones')
    st.text('Registros de cada estacion (Salida/llegada).')

    if df is not None and not df.empty:
        tipo_conteo = st.radio("Selecciona qué conteo deseas ver:", ["Salen", "Llegan"], horizontal=True)

        # Llamada a la función conteo_estacion con los datos de MiBici, nomenclatura y tipo de conteo
        conteo_df = conteo_estacion(df, nomenclatura_df, tipo_conteo)

        if conteo_df is not None and not conteo_df.empty:
            st.dataframe(conteo_df)
        else:
            st.warning('No hay datos disponibles para el conteo de estaciones')
    else:
        st.error("❌ No se pudo calcular el conteo debido a datos faltantes.")
    # =========== FIN CONTENIDO ==================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Apartado de nuevas columnas ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # =========== CONTENIDO ======================================
    st.divider()
    st.markdown('### Nuevas columnas')
    st.markdown('### Edades')
    st.image(io.imread(LOGO_PATH_AGE), width=800)
    opcion_edad=st.radio('Selecciona que opcion deseas mostrar en la tabla:',['Ninguno','Toda','Edad'], horizontal=True)
    mostrar_edad = st.toggle('Mostrar Toda la tabla')
    only_edad= st.toggle('Mostrar Unicamente la edad del usuario')
    #carga de datos
    df = load_cache(CACHE_FILE)

    if df is not None:
        df = edad(df)

        if opcion_edad == 'Toda':
            st.dataframe(df[["User_Id", "Gender","Age","Year_of_Birth","Trip_Start", "Trip_End", "Origin_Id", "Destination_Id"]])
        elif opcion_edad == 'Edad':
            st.dataframe(df[['User_Id', 'Age']])
        elif opcion_edad == 'Ninguna':
            return None
        
        if mostrar_edad:
            st.dataframe(df[["User_Id", "Gender","Age","Year_of_Birth","Trip_Start", "Trip_End", "Origin_Id", "Destination_Id"]])

        if only_edad:
            st.dataframe(df[['User_Id', 'Age']])

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
#===== Grafica Histograma === Hombres vs Mujeres = Uso de MiBici =
#=================================================================

#=================================================================
#===== Grafica Boxplot ==== Tiempo de viaje vs Ruta y genero =====
#=================================================================

#=================================================================
#===== Grafica Barras ==== Uso por Dias de la semana =============
#=================================================================

#=================================================================
#===== Grafico Correlacion === Uso de estanciones (Inicio / Fin) =
#=================================================================

#=================================================================
#===== Grafico Correlacion ==== Correlacion Dia de la semanas ====
#=================================================================

