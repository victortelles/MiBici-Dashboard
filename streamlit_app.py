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
#----- Configuracion Imagenes -------------------------
LOGO_PATH = r'./media/images/MiBici_Logo.png'
LOGO_PATH_AGE = r'./media/images/Edad.png'
IMAGE_PATH_STATION = r'./media/images/Estacion.png'
GRAPH_PATH = r'./media/images/grafico.png'
#----- Configuracion datos -------------------------
DATA_FOLDER = r'data/MiBici-Data'
DATA_NOMENCLATURA_FOLDER = r'data/Nomenclatura-Mibici-Data'
#----- Configuracion Cache -------------------------
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

#~~~~~ Funcion para calcular costo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calcular_costo(travel_time):
    '''Funcionalidad para calcular el costo de MiBici'''
    base_cost = 108
    extra_cost = 0

    total_minutes = travel_time.total_seconds() / 60
    if total_minutes > 30:
        extra_minutes = total_minutes - 30
        extra_cost += 29
        extra_minutes -= 30

        if extra_minutes > 0:
            extra_minutes += (extra_minutes // 30) * 40
            if extra_minutes % 30 > 0:
                extra_cost += 40

    return base_cost + extra_cost

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
    if 'Year_of_Birth' not in df.columns:
        st.error('❌ No se pudo calcular la edad porque la columna Year_of_Birth no se encuentra')
        return df

    df['Age'] = pd.to_numeric(df['Year_of_Birth'], errors='coerce') #Convertir numerico

    #Manejamiento de error de valores null
    if df['Year_of_Birth'].isnull().all():
        st.error('❌ No se puede calcular la edad porque todos los valores de "Year_of_Birth" son nulos.')
        return df

    # Funcion Calcular la edad
    df['Age'] = 2025 - df['Age']
    df.loc[(df['Age'] < 0) | (df['Age']> 150), 'Age'] = pd.NA # Filtrar valores erroneos (Limite excedido)
    return df

#----- Funcionalidad para crear columna de Tiempo recorrido -------
def tiempo_recorrido(df):
    '''Funcionalidad para añadir columna de "Travel_Time" calcular el tiempo de (Trip_Start y Trip_End)'''
    if df is None or df.empty:
        st.error('El DataFrame esta vacio o no es valido')
        return None

    try:
        #Convertir formato tiempo
        df['Trip_Start'] = pd.to_datetime(df['Trip_Start'])
        df['Trip_End'] = pd.to_datetime(df['Trip_End'])

        #Calculo para distancia
        df['Travel_Time'] = (df['Trip_End'] - df['Trip_Start'])

        #Formatear la diferencia en formato HH:MM:SS
        df['Travel_Time'] = df['Travel_Time'].apply(
            lambda x:str(x).split()[-1]
            if 'days' not in str(x) else(x) # Manejar diferencias > 24 H
        )

        #st.success('Columna "Travel_Time" agregada correctamente')
        return df

    except Exception as e:
        st.error('❌ No se pudo calcular el tiempo de recorrido porque la columna "')
        return None

#----- Funcionalidad para calcular la distancia de manhattan ------
def manhattan_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia de Manhattan en kilómetros."""
    lat_diff = abs(lat2 - lat1) * 111  # Aprox 111 km por grado de latitud
    lon_diff = abs(lon2 - lon1) * 111  # similar para la longitud
    return lat_diff + lon_diff

#----- Funcionalidad para crear un dataframe con distancia recorrida y tiempo
def distancia_tiempo(datos_filtrados, nomenclatura):
    """
    Funcionalidad para calcular la distancia recorrida.
    - Si el viaje está en diferentes estaciones, usa la distancia Manhattan.
    - Si inicia y termina en la misma estación, estima la distancia usando el tiempo de viaje.
    - Filtra viajes con duración menor a 1 minuto.
    """
    try:
        # Validar datos
        if datos_filtrados is None or datos_filtrados.empty:
            st.error('❌ No hay datos disponibles para calcular la distancia.')
            return None

        if nomenclatura is None or nomenclatura.empty:
            st.error('❌ No hay datos de nomenclatura disponibles.')
            return None

        # Unir nomenclatura para obtener latitud y longitud
        datos = datos_filtrados.merge(
            nomenclatura[['id', 'name', 'latitude', 'longitude']],
            left_on='Origin_Id', right_on='id', how='left'
        ).rename(columns={'name': 'Origin_Station', 'latitude': 'Origin_Lat', 'longitude': 'Origin_Lon'}).drop(columns=['id'])

        datos = datos.merge(
            nomenclatura[['id', 'name', 'latitude', 'longitude']],
            left_on='Destination_Id', right_on='id', how='left'
        ).rename(columns={'name': 'Destination_Station', 'latitude': 'Destination_Lat', 'longitude': 'Destination_Lon'}).drop(columns=['id'])

        # Calcular distancia de Manhattan
        datos['Distance_KM'] = datos.apply(lambda row: manhattan_distance(
            row['Origin_Lat'], row['Origin_Lon'], row['Destination_Lat'], row['Destination_Lon']
        ) if row['Origin_Id'] != row['Destination_Id'] else np.nan, axis=1)

        # Calcular tiempo de viaje en minutos
        datos['Trip_Start'] = pd.to_datetime(datos['Trip_Start'])
        datos['Trip_End'] = pd.to_datetime(datos['Trip_End'])
        datos['Duration_Min'] = (datos['Trip_End'] - datos['Trip_Start']).dt.total_seconds() / 60

        # Aproximar distancia si es la misma estación (pensando que: 150m/min como velocidad promedio)
        datos.loc[datos['Origin_Id'] == datos['Destination_Id'], 'Distance_KM'] = datos['Duration_Min'] * 0.15

        # Eliminar viajes con duración menor a 1 minuto
        datos = datos[datos['Duration_Min'] >= 1]

        # Redondear duración a minutos enteros
        datos['Duration_Min'] = datos['Duration_Min'].astype(int)

        # Seleccionar las columnas finales
        resultado = datos[['Trip_Id', 'Origin_Station', 'Destination_Station', 'Distance_KM', 'Duration_Min']]

        return resultado

    except Exception as e:
        st.error(f'❌ No se pudo calcular la distancia. Error: {str(e)}')
        return None

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Apartado de Graficos ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#===== Grafica Lineal ==== Cantidad de viajes * (Mes y año) ======
def graf_viaje(datos_filtrados, opcion_filtrado, year_selected, month_selected):
    ''' Grafica Lineal para contar la cantidad de viajes por mes y año'''
    try:
        # Validar datos no vengan vacios
        if datos_filtrados is None or datos_filtrados.empty:
            st.error('❌ No hay datos filtrados para generar la grafica')
            return

        # Opcion
        if opcion_filtrado == 'Año x Meses':
            # Agrupar por mes y contar la cantidad de viajes
            viajes_count = datos_filtrados.groupby('Month').size().reset_index(name='Cantidad_Viajes')
            # Configuracion Grafica
            titulo = f'Cantidad de viajes por Meses del (Año: {year_selected})'
            x_label = 'Mes'
            x_values = viajes_count['Month']
        else:
            # Agrupar por año y contar la cantidad de viajes
            viajes_count = datos_filtrados.groupby('Year').size().reset_index(name='Cantidad_Viajes')
            # Configuracion Grafica
            titulo = f'Cantidad de viajes por Años del (Mes: {month_selected})'
            x_label = 'Año'
            x_values = viajes_count['Year']

        # Mostrar Datos de conteo de viajes
        st.markdown('#### 📊 Conteo de Viajes:')
        st.dataframe(viajes_count)

        #creacion de la grafica
        plt.figure(figsize=(10,6))
        sns.lineplot(data=viajes_count, x=x_values, y='Cantidad_Viajes', marker='o')
        plt.title(titulo, fontsize=16)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel('Cantidad de Viajes', fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        #Mostrar grafico
        return st.pyplot(plt)
    except Exception as e:
        st.error('❌ No se pudo generar la gráfica de viajes')

#===== Grafica Barras ==== Promedio de viaje (dia * semana) ======
def graf_uso_semanal(datos_filtrados, opcion_filtrado, year_selected, month_selected):
    ''' Grafica para contar el uso de MiBici por semana'''
    try:
        #Validar datos no esten vacios
        if datos_filtrados is None or datos_filtrados.empty:
            st.error('❌ No hay datos filtrados para generar la grafica')
            return
        #Convertir a datetime
        datos_filtrados['Trip_Start'] = pd.to_datetime(datos_filtrados['Trip_Start'])
        #Obtener el dia de la semana (0 = Lunes - 6 = Domingo)
        datos_filtrados['Day_Week'] = datos_filtrados['Trip_Start'].dt.dayofweek
        #Diccionario para mapear el # a dias
        dias_semana = {0: 'Lunes', 1: 'Martes', 2: 'Miercoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sabado', 6: 'Domingo',}
        #Remplazar los valores numericos a nombre del dia
        datos_filtrados['Day_Week'] = datos_filtrados['Day_Week'].map(dias_semana)
        #Conteo de # viajes por cada dia de la semana
        count_viajes = datos_filtrados['Day_Week'].value_counts().reindex(dias_semana.values(), fill_value = 0)

        #Opcion filtrado [Año x Meses | Mes x Año]
        if opcion_filtrado == "Año x Meses":
            titulo = f'Uso de MiBici por Día de la Semana - Año {year_selected}'
            subtitulo = f'Meses seleccionados: {month_selected}'
        elif opcion_filtrado == "Mes x Años":
            titulo = f'Uso de MiBici por Día de la Semana - Mes {month_selected}'
            subtitulo = f'Años seleccionados: {year_selected}'
        else:
            titulo = 'Uso de MiBici por Día de la Semana'
            subtitulo = ''

        # Mostrar tabla de conteo
        st.markdown(f'#### 📊 {titulo}:')
        if subtitulo:
            st.markdown(f'{subtitulo}')
        st.dataframe(count_viajes.reset_index().rename(columns={'index': 'Day', 'Day_Week': 'Cantidad de Viajes'}))

        #Creacion y Configuracion Grafica
        plt.figure(figsize=(10,6))
        sns.barplot(x=count_viajes.index, y =count_viajes.values, hue=count_viajes.index, palette='coolwarm', legend=False)

        # conf.
        plt.title(f'{titulo}', fontsize=16)
        plt.xlabel('Dia de la semmana', fontsize=12)
        plt.ylabel('Cantidad de Viajes', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        #Mostrar grafico
        st.pyplot(plt)

    except Exception as e:
        st.error(f'❌ No se pudo generar la gráfica de uso semanal {e}')

#===== Grafica Barras ==== Uso por Dias de MiBici ================
def graf_dias(datos_filtrados, opcion_filtrado, year_selected, month_selected):
    '''Grafica para mostrar el uso de MiBici en los dias.'''
    try:
        #Validar que los datos no esten vacios
        if datos_filtrados is None or datos_filtrados.empty:
            st.error('❌ No hay datos filtrados para generar la grafica.')
            return

        #Extraer el dia del mes
        datos_filtrados['Day'] = datos_filtrados['Trip_Start'].dt.day

        #Contar Viajes por dia y aplicar filtros
        if opcion_filtrado == 'Año x Meses':
            #Agrupar por Año-Mes-Dia
            count_days = datos_filtrados.groupby(['Year', 'Month', 'Day']).size().reset_index(name='Count')
            # Convertir Month a string para visualización clara
            count_days['Month'] = count_days['Month'].astype(str)

        else:
            # Agrupar por Año-Mes-Día
            count_days = datos_filtrados.groupby(['Year', 'Month', 'Day']).size().reset_index(name='Count')
            # Convertir Year a string para visualizarlo correctamente
            count_days['Year'] = count_days['Year'].astype(str)

        st.markdown('#### 📊 Conteo de Viajes por Día:')
        st.dataframe(count_days)

        # Creacion y conf la grafica
        plt.figure(figsize=(12,6))
        sns.set_style('whitegrid')

        if opcion_filtrado == 'Año x Meses':
            # Grafico varios meses del mismo año
            sns.lineplot(data=count_days, x = 'Day', y='Count', hue='Month', markers='o', palette='tab10')
            plt.title(f'Cantidad de Viajes por Día - Año: {year_selected}', fontsize=14)
            plt.xlabel('Día del Mes', fontsize=12)
            plt.ylabel('Cantidad de Viajes', fontsize=12)
            plt.legend(title="Meses")

        else:
            # Graficar múltiples años del mismo mes con distintos colores
            sns.lineplot(data=count_days, x='Day', y='Count', hue='Year', marker='o', palette='tab10')
            plt.title(f'Cantidad de Viajes por Día en el Mes: {month_selected}', fontsize=14)
            plt.xlabel('Día del Mes', fontsize=12)
            plt.ylabel('Cantidad de Viajes', fontsize=12)
            plt.legend(title="Años")

        # Mostrar la gráfica en Streamlit
        st.pyplot(plt)

    except Exception as e:
        st.error(f'❌ No se pudo generar la grafica {str(e)}')

#===== Grafica Histograma === Hombres vs Mujeres = Uso de MiBici =
def graf_gender_versus(datos_filtrados, opcion_filtrado, year_selected, month_selected):
    '''Grafica para mostrar la comparativa de H vs M al usar MiBici durante la semana'''
    try:
        #Validar que los datos no esten
        if datos_filtrados is None or datos_filtrados.empty:
            st.error('❌ No hay datos filtrados para generar la grafica')
            return
        # Convertir a Trip_Start a datetime
        datos_filtrados['Trip_Start'] = pd.to_datetime(datos_filtrados['Trip_Start'])
        #Obtener el dia de la semana (0 = Lunes - 6 = Domingo)
        datos_filtrados['Day_Week'] = datos_filtrados['Trip_Start'].dt.dayofweek
        #Diccionario para mapear el # a dias
        dias_semana = {0: 'Lunes', 1: 'Martes', 2: 'Miercoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sabado', 6: 'Domingo',}
        #Remplazar los valores numericos a nombre del dia
        datos_filtrados['Day_Week'] = datos_filtrados['Day_Week'].map(dias_semana)
        #Definir el orden correcto de los dias de la semana
        orden_dias = ['Lunes','Martes','Miercoles','Jueves', 'Viernes','Sabado','Domingo',]
        datos_filtrados['Day_Week'] = pd.Categorical(datos_filtrados['Day_Week'],categories = orden_dias, ordered=True)
        #Conteo de # viajes por genero durante la semana dia de la semana
        count_gender = datos_filtrados.groupby(['Day_Week', 'Gender'], observed=False).size().reset_index(name='Cantidad_Viajes')

        # Pivotear la tabla para que cada género sea una columna
        count_pivot = count_gender.pivot(index='Day_Week', columns='Gender', values='Cantidad_Viajes').fillna(0)

        #Opcion filtrado [Año x Meses | Mes x Año]
        if opcion_filtrado == "Año x Meses":
            titulo = f'Uso de MiBici por Día de la Semana - Año {year_selected}'
            subtitulo = f'Meses seleccionados: {month_selected}'
        elif opcion_filtrado == "Mes x Años":
            titulo = f'Uso de MiBici por Día de la Semana - Mes {month_selected}'
            subtitulo = f'Años seleccionados: {year_selected}'
        else:
            titulo = 'Uso de MiBici por Día de la Semana'
            subtitulo = ''

        # Mostrar la tabla
        st.markdown(f'#### 📊 {titulo}')
        if subtitulo:
            st.markdown(f'{subtitulo}')
        st.dataframe(count_pivot)

        # Crear la gráfica
        plt.figure(figsize=(10,6))
        colores = ['#ff69b4', '#1f77b4']
        count_pivot.plot(kind='bar', stacked=False, color=colores, alpha=0.8, width=0.8)

        # Configuración del gráfico
        plt.title(f'Comparativa H vs M de {titulo} ', fontsize=16)
        plt.xlabel('Día de la Semana', fontsize=12)
        plt.ylabel('Cantidad de Viajes', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Genero', labels=['Mujeres', 'Hombres'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Mostrar gráfico en Streamlit
        st.pyplot(plt)

        # Creacion gráfico de pastel
        st.markdown("#### 🎯 Proporción de Viajes por Género")
        total_por_genero = datos_filtrados['Gender'].value_counts()
        colores_pastel = ['#1f77b4', '#ff69b4']

        plt.figure(figsize=(6,6))
        plt.pie(total_por_genero, labels=total_por_genero.index, autopct='%1.1f%%', colors=colores_pastel, startangle=90, wedgeprops={'edgecolor': 'white'})
        plt.title("Distribución de Viajes por Género")

        # Mostrar gráfico de pastel en Streamlit
        st.pyplot(plt)

    except Exception as e:
        st.error(f'❌ No se pudo generar la grafica')

#===== Grafico Correlacion ==== Correlacion Dia de la semanas ====
def graf_dia_time(datos_filtrados):
    try:
        #Validar que los datos no esten
        if datos_filtrados is None or datos_filtrados.empty:
            st.error('❌ No hay datos filtrados para generar la grafica')
            return

        # Llamar a la funcion para extraer el 'Travel_Time'
        datos_filtrados = tiempo_recorrido(datos_filtrados)

        # Validacion de que exista la columna "Travel_Time"
        if 'Travel_Time' not in datos_filtrados.columns:
            st.error('❌ La columna Travel_Time No existe, no se puede crear la funcion')
            return

        # Debbugin
        #st.write("Primeros 5 valores de Travel_Time:")
        #st.write(datos_filtrados[['Trip_Id', 'Trip_Start', 'Trip_End', 'Travel_Time']].head(15))

        #Convertir 'Travel_time' a timedelta
        if not pd.api.types.is_timedelta64_dtype(datos_filtrados['Travel_Time']):
            datos_filtrados['Travel_Time'] = pd.to_timedelta(datos_filtrados['Travel_Time'])

        #Diccionario para mapear el # a dias
        dias_semana = {0: 'Lunes', 1: 'Martes', 2: 'Miercoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sabado', 6: 'Domingo',}
        datos_filtrados['Day_Week'] = datos_filtrados['Trip_Start'].dt.weekday.map(dias_semana)

#        #Calcular promedio de tiempo de viaje por dia de la semana
#        promedio_viajes = datos_filtrados.groupby('Day_Week')['Travel_Time'].mean().reset_index()
#        promedio_viajes['Travel_Time'] = promedio_viajes['Travel_Time'].dt.total_seconds() / 60  # Convertir a minutos

        #Ordenar dias de la semana
        orden_dias = ['Lunes','Martes','Miercoles','Jueves', 'Viernes','Sabado','Domingo',]
        datos_filtrados['Day_Week'] = pd.Categorical(datos_filtrados['Day_Week'],categories = orden_dias, ordered=True)
        datos_filtrados = datos_filtrados.sort_values('Day_Week')

        #Convertir Travel_Time a minutos
        datos_filtrados['Travel_Time_Minutos'] = datos_filtrados['Travel_Time'].dt.total_seconds() / 60

        #Calcular los percentiles
        y_min = datos_filtrados['Travel_Time_Minutos'].quantile(0.0) #percentil 0
        y_max = datos_filtrados['Travel_Time_Minutos'].quantile(0.99) #percentil 99

        # --- Gráfico de Dispersión con Línea de Tendencia ---
        fig, ax = plt.subplots(figsize=(10, 5))

        # Mapear los días de la semana a números para el eje X
        datos_filtrados['Day_Week_Num'] = datos_filtrados['Day_Week'].map({dia: i for i, dia in enumerate(orden_dias)})

        # Scatter plot
        sns.scatterplot(
            x='Day_Week_Num', 
            y='Travel_Time_Minutos', 
            data=datos_filtrados, 
            color='blue', 
            alpha=0.6, 
            ax=ax
        )

        # Línea de tendencia
        sns.regplot(
            x='Day_Week_Num', 
            y='Travel_Time_Minutos', 
            data=datos_filtrados, 
            scatter=False, 
            color='red', 
            ax=ax
        )
        
        #calcular el promedio de "Travel_Time_Minutos" por dia de la semana
        promedio_viajes = datos_filtrados.groupby('Day_Week')['Travel_Time_Minutos'].mean().reset_index()

        # Mostrar el promedio como una línea horizontal
        for dia, promedio in zip(promedio_viajes['Day_Week'], promedio_viajes['Travel_Time_Minutos']):
            ax.axhline(promedio, color='green', linestyle='--', alpha=0.5, label=f'Promedio {dia}: {promedio:.2f} min')

        # Configuración del gráfico
        ax.set_title("Correlación Día de la Semana - Tiempo de Viaje", fontsize=16)
        ax.set_xlabel("Día de la Semana", fontsize=12)
        ax.set_ylabel("Tiempo de Viaje (minutos)", fontsize=12)
        ax.set_xticks(range(len(orden_dias)))
        ax.set_xticklabels(orden_dias, rotation=45)
        ax.set_ylim(y_min, y_max)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)

        # --- Boxplot para ver distribución ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(
            [datos_filtrados[datos_filtrados['Day_Week'] == dia]['Travel_Time_Minutos'] for dia in orden_dias],
            labels=orden_dias
        )
        ax.set_title("Distribución del Tiempo de Viaje por Día de la Semana", fontsize=16)
        ax.set_xlabel("Día de la Semana", fontsize=12)
        ax.set_ylabel("Tiempo de Viaje (minutos)", fontsize=12)
        ax.set_ylim(y_min, y_max)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)


    except Exception as e:
        st.error(f'❌ No se pudo generar la grafica {str(e)}')

def graf_dia_mes_time(datos_filtrados):
    try:
        # Validar que los datos no estén vacíos
        if datos_filtrados is None or datos_filtrados.empty:
            st.error('❌ No hay datos filtrados para generar la gráfica.')
            return

        # Llamar a la función para calcular la columna Travel_Time
        datos_filtrados = tiempo_recorrido(datos_filtrados)

        # Verificar si la columna Travel_Time se creó correctamente
        if 'Travel_Time' not in datos_filtrados.columns:
            st.error('❌ La columna "Travel_Time" no se pudo crear.')
            return

        # Convertir 'Travel_Time' a timedelta si no lo es
        if not pd.api.types.is_timedelta64_dtype(datos_filtrados['Travel_Time']):
            datos_filtrados['Travel_Time'] = pd.to_timedelta(datos_filtrados['Travel_Time'])

        # Extraer el día del mes de Trip_Start
        datos_filtrados['Day_Month'] = datos_filtrados['Trip_Start'].dt.day

        # Convertir Travel_Time a minutos
        datos_filtrados['Travel_Time_Minutos'] = datos_filtrados['Travel_Time'].dt.total_seconds() / 60

        # --- Opciones de Filtrado ---
        st.markdown('### Filtrado por Tiempo de Viaje')
        opcion_filtro = st.radio(
            'Selecciona una opción de filtrado:',
            ['Mostrar más de 1 minuto', 'Mostrar más de 15 minutos']
        )

        # Aplicar el filtro según la opción seleccionada
        if opcion_filtro == 'Mostrar más de 1 minuto':
            datos_filtrados = datos_filtrados[datos_filtrados['Travel_Time_Minutos'] >= 1]  # 1 minuto o más
        elif opcion_filtro == 'Mostrar más de 15 minutos':
            datos_filtrados = datos_filtrados[datos_filtrados['Travel_Time_Minutos'] >= 15]  # 15 minutos o más

        # --- Gráfico de Dispersión con Línea de Tendencia ---
        fig, ax = plt.subplots(figsize=(10, 5))

        # Debug: Verificar los valores de Travel_Time_Minutos
        st.write("Valores de Travel_Time_Minutos:")
        st.write(datos_filtrados['Travel_Time_Minutos'].describe())

        # Calcular los percentiles 5 y 95 para ajustar el eje y
        y_min = datos_filtrados['Travel_Time_Minutos'].quantile(0.05)  # Percentil 5
        y_max = datos_filtrados['Travel_Time_Minutos'].quantile(0.95)  # Percentil 95

        # Scatter plot
        sns.scatterplot(
            x='Day_Month', 
            y='Travel_Time_Minutos',  # Usar minutos directamente
            data=datos_filtrados, 
            color='blue', 
            alpha=0.6, 
            ax=ax
        )

        # Línea de tendencia
        sns.regplot(
            x='Day_Month', 
            y='Travel_Time_Minutos',  # Usar minutos directamente
            data=datos_filtrados, 
            scatter=False, 
            color='red', 
            ax=ax
        )

        # Configuración del gráfico
        ax.set_title(f"Correlación Día del Mes - Tiempo de Viaje ({opcion_filtro})", fontsize=16)
        ax.set_xlabel("Día del Mes", fontsize=12)
        ax.set_ylabel("Tiempo de Viaje (minutos)", fontsize=12)
        ax.set_xticks(range(1, 32))  # Días del mes (1-31)
        ax.set_ylim(y_min, y_max)  # Ajustar el eje y basado en percentiles
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)

        # --- Boxplot para ver distribución ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(
            [datos_filtrados[datos_filtrados['Day_Month'] == dia]['Travel_Time_Minutos'] for dia in range(1, 32)],
            labels=range(1, 32)
        )
        ax.set_title(f"Distribución del Tiempo de Viaje por Día del Mes ({opcion_filtro})", fontsize=16)
        ax.set_xlabel("Día del Mes", fontsize=12)
        ax.set_ylabel("Tiempo de Viaje (minutos)", fontsize=12)
        ax.set_ylim(y_min, y_max)  # Ajustar el eje y basado en percentiles
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.error(f'❌ No se pudo generar la gráfica. Error: {str(e)}')


#===== Grafica Correlacion ==== Edad - Tiempo promedio ==========
def graf_edad_time(datos_filtrados, opcion_filtrado, year_selected, month_selected):
    '''Grafica para mostrar la edad con tiempo promedio'''
    try:
        # Validar que los datos no estén vacíos
        if datos_filtrados is None or datos_filtrados.empty:
            st.error('❌ No hay datos filtrados para generar la gráfica.')
            return

        #Calcular edad usuarios
        datos_filtrados = edad(datos_filtrados)

        #Calcular tiempo de viaje
        datos_filtrados = tiempo_recorrido(datos_filtrados)

        #Verificar que las columnas necesarias esten presentes
        if 'Age' not in datos_filtrados.columns or 'Travel_Time' not in datos_filtrados.columns:
            st.error('❌ No se pudieron calcular las columnas necesarias (Age o Travel_Time).')
            return

        #convertir Travel_Time a minutos
        datos_filtrados['Travel_Time'] = pd.to_timedelta(datos_filtrados['Travel_Time'])
        datos_filtrados['Travel_Time_Minutos'] = datos_filtrados['Travel_Time'].dt.total_seconds() / 60

        #Agrupar por edad y calcular el tiempo promedio de viaje
        tiempo_promedio_edad = datos_filtrados.groupby('Age')['Travel_Time_Minutos'].mean().reset_index()

        #Filtrar edades validas
        tiempo_promedio_edad = tiempo_promedio_edad[(tiempo_promedio_edad['Age'] >= 16) & (tiempo_promedio_edad['Age'] <= 120)]
        #Ajustar el Eje y
        y_max = tiempo_promedio_edad['Travel_Time_Minutos'].quantile(0.99) #percentil 99

        #Opcion filtrado [Año x Meses | Mes x Año]
        if opcion_filtrado == "Año x Meses":
            titulo = f'Correlacion Edad - Tiempo promedio de viaje. - Año {year_selected}'
            subtitulo = f'Meses seleccionados: {month_selected}'
        elif opcion_filtrado == "Mes x Años":
            titulo = f'Correlacion Edad - Tiempo promedio de viaje. - Mes {month_selected}'
            subtitulo = f'Años seleccionados: {year_selected}'
        else:
            titulo = 'Correlacion Edad - Tiempo promedio de viaje.'
            subtitulo = ''

        # Mostrar la tabla
        st.markdown(f'#### 📊 {titulo}')
        if subtitulo:
            st.markdown(f'{subtitulo}')
        st.dataframe(tiempo_promedio_edad)

        #Creacion de grafico
        fig, ax = plt.subplots(figsize = (12, 6))

        #ScatterPlot
        sns.scatterplot(
            x="Age",
            y="Travel_Time_Minutos",
            data=tiempo_promedio_edad,
            color='blue',
            alpha=0.6,
            ax=ax
        )

        #Linea de tendencia
        sns.regplot(
            x="Age",
            y="Travel_Time_Minutos",
            data=tiempo_promedio_edad,
            scatter=False,
            color='red',
            ax=ax
        )

        # conf
        ax.set_title(f'{titulo}', fontsize =16)
        ax.set_xlabel('Edad (>= 16) ', fontsize=12)
        ax.set_ylabel('Tiempo Promedio de viaje (Minutos)', fontsize=12)
        ax.set_ylim(0, y_max)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        #Mostrar grafico
        st.pyplot(fig)

    except Exception as e:
        st.error(f'❌ No se pudo generar la gráfica. Error: {str(e)}')

#===== Grafica pastel ==== % Uso MiBici - Edad ===================
def graf_edad_pastel(datos_filtrados):
    '''Gráfica de pastel para mostrar el porcentaje de uso de MiBici por rango de edad'''
    try:
        # Validar que los datos no estén vacíos
        if datos_filtrados is None or datos_filtrados.empty:
            st.error('❌ No hay datos filtrados para generar la gráfica.')
            return

        # Calcular edad usuarios
        datos_filtrados = edad(datos_filtrados)

        # Verificar que la columna 'Age' está presente
        if 'Age' not in datos_filtrados.columns:
            st.error('❌ No se pudo calcular la edad de los usuarios.')
            return

        # Filtrar edades válidas (16 a 120 años)
        datos_filtrados = datos_filtrados[(datos_filtrados['Age'] >= 16) & (datos_filtrados['Age'] <= 120)]

        # Definir rangos de edad
        bins = [16, 25, 35, 45, 55, 65, 75, 120]
        labels = ['16-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
        datos_filtrados['Rango_Edad'] = pd.cut(datos_filtrados['Age'], bins=bins, labels=labels, right=False)

        # Contar cantidad de usuarios por rango de edad
        edad_counts = datos_filtrados['Rango_Edad'].value_counts().sort_index()

        # Crear gráfica de pastel
        fig, ax = plt.subplots(figsize=(8, 8))
        colores = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFD700', '#FF69B4', '#B2B2B2']
        ax.pie(
            edad_counts,
            labels=edad_counts.index,
            autopct='%1.1f%%',
            colors=colores,
            startangle=140,
            wedgeprops={'edgecolor': 'black'}
        )
        ax.set_title('Distribución del Uso de MiBici por Rango de Edad')

        # Mostrar gráfica
        st.pyplot(fig)

    except Exception as e:
        st.error(f'❌ No se pudo generar la gráfica. Error: {str(e)}')

#===== Grafica Barras === Uso de estaciones  ====================
def graf_use_station(datos_filtrados, nomenclatura_df, tipo):
    '''Grafica para mostrar el uso de cada estacion'''
    try:
        # Validar que los datos no estén vacíos
        if datos_filtrados is None or datos_filtrados.empty:
            st.error('❌ No hay datos filtrados para generar la gráfica.')
            return

        #Obtener el conteo de viajes por estacion
        count = conteo_estacion(datos_filtrados, nomenclatura_df, tipo)

        # Validar que el conteo no se encuentre vacio
        if count is None or count.empty:
            st.error('❌ No se pudo generar el conteo de estaciones.')
            return

        #Extraer el identificador de c/Estacion
        if tipo == 'Salen':
            count['Station_Code'] = count['Origin_Station'].str.extract(r'\((.*?)\)')
            x_label = 'Estacion de Salida'
            y_label = 'Conteo de Viajes de Salida'
        elif tipo == 'Llegan':
            count['Station_Code'] = count['Destination_Station'].str.extract(r'\((.*?)\)')
            x_label = 'Estacion de Llegada'
            y_label = 'Conteo de Viajes de Llegada'
        else:
            st.error('❌ Tipo de conteo no valido. Usa "Salen" o "Llegan".')
            return

        #Ordenar por conteo de viajes
        count = count.sort_values(by=count.columns[2], ascending= False)

        # Creacion de grafico y configuracion

        # --- Grafico de barras ---
        plt.figure(figsize=(12,6))
        sns.barplot(
            x = 'Station_Code',
            y = count.columns[2],
            data = count,
            palette='coolwarm'
        )

        # conf grafico
        plt.title(f' Uso de Estaciones ({tipo})', fontsize=16)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        #Mostrar grafico
        st.pyplot(plt)

    except Exception as e:
        st.error(f'❌ No se pudo generar la gráfica. Error: {str(e)}')

#===== Grafica barras ==== Total de dinero gastado (aproximado) =
def graf_money(datos_filtrados, opcion_filtrado, year_selected, month_selected):
    '''Grafica para mostrar el total de dinero gastado'''
    try:
        # Validar que los datos no estén vacíos
        if datos_filtrados is None or datos_filtrados.empty:
            st.error('❌ No hay datos filtrados para generar la gráfica.')
            return

        # Convertir Trip_Start a datetime si no está en formato datetime
        datos_filtrados['Trip_Start'] = pd.to_datetime(datos_filtrados['Trip_Start'], errors='coerce')

        # Extraer el día de la semana en español
        dias_semana = {
            'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miercoles',
            'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sabado', 'Sunday': 'Domingo'
        }
        datos_filtrados['Day'] = datos_filtrados['Trip_Start'].dt.day_name().map(dias_semana)

        # Convertir Travel_Time a minutos si es timedelta
        if pd.api.types.is_timedelta64_dtype(datos_filtrados['Travel_Time']):
            datos_filtrados['Travel_Time_Minutos'] = datos_filtrados['Travel_Time'].dt.total_seconds() / 60
        else:
            st.error('❌ Error: Travel_Time no está en formato timedelta.')
            return

        # Calcular costo
        datos_filtrados['Cost'] = datos_filtrados['Travel_Time'].apply(calcular_costo)

        # Agrupar por día de la semana y sumar el costo total
        costo_por_dia = datos_filtrados.groupby('Day', observed=True)['Cost'].sum().reset_index()

        # Ordenar días de la semana correctamente
        dias_ordenados = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
        costo_por_dia['Day'] = pd.Categorical(costo_por_dia['Day'], categories=dias_ordenados, ordered=True)
        costo_por_dia = costo_por_dia.sort_values('Day')

        #Opcion filtrado [Año x Meses | Mes x Año]
        if opcion_filtrado == "Año x Meses":
            titulo = f'Gasto total por dia de la semana - Año {year_selected}'
            subtitulo = f'Meses seleccionados: {month_selected}'
        elif opcion_filtrado == "Mes x Años":
            titulo = f'Gasto total por dia de la semana - Mes {month_selected}'
            subtitulo = f'Años seleccionados: {year_selected}'
        else:
            titulo = 'Gasto total por dia de la semana'
            subtitulo = ''

        # Mostrar la tabla
        st.markdown(f'#### 📊 {titulo}')
        if subtitulo:
            st.markdown(f'{subtitulo}')
        st.dataframe(costo_por_dia)

        # --- Gráfico de barras ---
        plt.figure(figsize=(10, 5))
        sns.barplot(
            x='Day',
            y='Cost',
            data=costo_por_dia,
            palette='summer'
        )

        # Configuración del gráfico
        plt.title(f'{titulo}', fontsize=14)
        plt.xlabel('Día de la semana', fontsize=12)
        plt.ylabel('Costo Total (MXN) Gastado', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Mostrar gráfico
        st.pyplot(plt)

    except Exception as e:
        st.error(f'❌ No se pudo generar la gráfica. Error: {str(e)}')

#===== Grafica Pastel ==== % estaciones funcionales o no func. ==
def graf_estacion_func(nomenclatura_df):
    """gráfico de pastel para visualizar la distribución de categorias de estaciones de MiBici."""
    try:
        # Verificar si el dataframe es válido
        if nomenclatura_df is None or nomenclatura_df.empty:
            st.warning('⚠ No hay datos disponibles para generar la gráfica.')
            return

        # Verificar que la columna "status" exista en el dataframe
        if "status" not in nomenclatura_df.columns:
            st.error('❌ La columna "status" no existe en el dataframe.')
            return

        # Contar cada tipo de estado
        conteo_status = nomenclatura_df["status"].value_counts()

        # Verificar si hay datos suficientes para graficar
        if conteo_status.empty:
            st.warning('⚠ No hay estados suficientes para generar la gráfica.')
            return

        # Configurar la figura
        plt.figure(figsize=(8, 6))

        # Crear el gráfico de pastel
        plt.pie(
            conteo_status,
            labels=conteo_status.index,
            autopct='%1.1f%%',
            colors=sns.color_palette("pastel"),
            startangle=170,
            wedgeprops={'edgecolor': 'black'}
        )

        plt.title('Distribución de Estaciones en Servicio')
        plt.axis('equal')  # Asegura que el gráfico sea un círculo

        # Mostrar en Streamlit
        st.pyplot(plt)

    except Exception as e:
        st.error(f'❌ Error al generar la gráfica: {str(e)}')


#~~~~~~~~~~~~~~~~~~~~~~~~~ Interfaz APP ~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    #----- Configuracion inicial --------------------------------
    st.image(io.imread(LOGO_PATH), width=200)
    st.title('Datos de mi Bici (2014-2024)')
    st.subheader(':blue[Indice]')
    st.markdown('[Datos Generales](#secc_datos_generales)')
    st.markdown('[Estaciones](#secc_estaciones)')
    st.markdown('[Nomenclatura](#secc_nomenclatura)')
    st.markdown('[Distancia Recorrida](#secc_distancia_recorrida)')
    st.markdown('[Tiempos promedio](#secc_promedio_viaje)')
    st.markdown('[Edades](#secc_edades)')
    st.markdown('[Graficos](#secc_graficos)')
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
            st.sidebar.write('⚠ Escoge maximo 4')
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
            st.sidebar.write('⚠ Escoge maximo 4')
            #Aplicacion filtro
            datos_filtrados = df[(df['Month'] == month_selected) & (df['Year'].isin(year_selected))]
    # =========== FIN SIDEBAR ====================================

        #Mostrando y aplicando filtros
        st.header(f'Mostrando datos:', anchor='secc_datos_generales')
        st.markdown(f'### Año: {year_selected}')
        st.markdown(f'### Meses: {month_selected}')
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
    st.sidebar.header('Mostrar Nomenclatura' , anchor='secc_nomenclatura')
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
    st.header('Datos de estaciones', anchor='secc_estaciones')
    st.image(io.imread(IMAGE_PATH_STATION), width=800)
    st.text('Visualización de las estaciones con sus respectivas agrupaciones.')

    #Validar si hay datos en los df.
    if datos_filtrados is not None and nomenclatura_df is not None:
        estaciones_df = estaciones(datos_filtrados, nomenclatura_df)
        if estaciones_df is not None and not estaciones_df.empty:
            st.dataframe(estaciones_df)
        else:
            st.warning('No se encontraron datos de estaciones')
    else:
        st.error('❌ No se pudo calcular estaciones debido a datos faltantes.')

    #~~~~~ Contador de viajes por estación ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.divider()
    st.header('Contador de estaciones', anchor='contador_Estaciones')
    st.text('Registros de cada estacion (Salida/llegada).')

    if datos_filtrados is not None and not df.empty:
        tipo_conteo = st.radio("Selecciona qué conteo deseas ver:", ["Salen", "Llegan"], horizontal=True)

        #Redireccion al grafico
        st.markdown('[Pulsa para ver el Grafico de uso por estacion](#grafico_Estaciones)')

        # Llamada a la función conteo_estacion con los datos de MiBici, nomenclatura y tipo de conteo
        conteo_df = conteo_estacion(datos_filtrados, nomenclatura_df, tipo_conteo)

        if conteo_df is not None and not conteo_df.empty:
            st.dataframe(conteo_df)
        else:
            st.warning('No hay datos disponibles para el conteo de estaciones')
    else:
        st.error("❌ No se pudo calcular el conteo debido a datos faltantes.")

    #~~~~~ Distancia recorrida ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Llamar a la función con los datos filtrados
    df_distancia = distancia_tiempo(datos_filtrados, nomenclatura_df)

    # Mostrar la tabla en Streamlit si hay datos
    if df_distancia is not None and not df_distancia.empty:
        st.header('Distancia Recorrida (Aproximacion)', anchor='secc_distancia_recorrida')
        st.dataframe(df_distancia)
    else:
        st.warning('⚠ No hay datos disponibles para mostrar.')

    # =========== FIN CONTENIDO ==================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~ Apartado de nuevas columnas ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # =========== CONTENIDO ======================================
    st.divider()
    st.markdown('### Nuevas columnas')
    #~~~~~ Apartado mostrar edades ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.header('Edades', anchor='secc_edades')
    st.image(io.imread(LOGO_PATH_AGE), width=800)
    opcion_edad=st.radio('Selecciona que opcion deseas mostrar en la tabla:',['Edad', 'Toda'], horizontal=True)

    try:
        df = load_cache(CACHE_FILE)
        if df is not None and not df.empty:
            #Llamando la funcion
            datos_filtrados = edad(datos_filtrados)

            if opcion_edad == 'Toda':
                st.dataframe(datos_filtrados[["User_Id","Age","Gender","Year_of_Birth","Trip_Start", "Trip_End", "Origin_Id", "Destination_Id"]])
            elif opcion_edad == 'Edad':
                st.dataframe(datos_filtrados[['User_Id', 'Age','Gender']])

            else:
                st.error('❌ No se pudo calcular edades debido a datos faltantes.')

    except Exception as e:
        st.error(f'❌ No se pudo cargar el archivo de datos. Error: {e}')
        return

    # =========== CONTENIDO GRAFICOS ===============================================================================
    st.divider()
    st.header('Graficos', anchor='secc_graficos')
    st.image(io.imread(GRAPH_PATH), width=600)

    #st.markdown('### Gráfica de Viajes por Mes y Año')
    if datos_filtrados is not None and not datos_filtrados.empty:
        #----- Llamada a funcion de graficos ---------------------
        st.markdown('### Viajes por Mes y Año')
        graf_viaje(datos_filtrados, opcion_filtrado, year_selected, month_selected)
        st.markdown('### Uso de MiBici en durante la semana')
        graf_uso_semanal(datos_filtrados, opcion_filtrado, year_selected, month_selected)
        st.markdown('### Uso de MiBici en dias')
        graf_dias(datos_filtrados, opcion_filtrado, year_selected, month_selected)
        st.markdown('### % Estaciones funcionales')
        graf_estacion_func(nomenclatura_df)
        st.markdown('### Comparativa Hombres vs Mujeres en Uso de MiBici en los dias de la semana')
        graf_gender_versus(datos_filtrados, opcion_filtrado, year_selected, month_selected)
        st.markdown('### Correlacion Edad - Tiempo Promedio')
        graf_edad_time(datos_filtrados, opcion_filtrado, year_selected, month_selected)
        st.markdown('### Pastel Edad - Rango de edades')
        graf_edad_pastel(datos_filtrados)
        st.header('Uso de estaciones.', anchor='grafico_Estaciones')
        st.markdown('Presione aqui, para cambiar la opcion [Salen/Llegada](#contador_Estaciones)')
        graf_use_station(datos_filtrados, nomenclatura_df, tipo_conteo)
        st.markdown('### Gasto total de MiBici ')
        graf_money(datos_filtrados, opcion_filtrado, year_selected, month_selected)
        st.markdown('### Grafico correlacion')
        graf_dia_time(datos_filtrados)
        st.markdown('### Correlación Día del Mes - Tiempo de Viaje')
        graf_dia_mes_time(datos_filtrados)
    else:
        st.write("No hay datos filtrados para mostrar la grafica.")
    # =========== FIN CONTENIDO GRAFICOS ===========================================================================


    #~~~~~ Apartado Calcular Tiempo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.divider()
    st.markdown('### Tiempo Recorrido')
    #st.image(io.imread(LOGO_PATH_AGE), width=800)
    opcion_time = st.toggle('Selecciona esta opcion si deseas mostrar en tabla el tiempo recorrido:',value=False)

    try:
        df = load_cache(CACHE_FILE)
        if df is not None and not df.empty:
            #Llamando la funcion
            datos_filtrados = tiempo_recorrido(datos_filtrados)

            if opcion_time == True:
                st.dataframe(datos_filtrados[['Trip_Id', 'User_Id', 'Gender', 'Age', 'Travel_Time']])
            elif opcion_time == False:
                return None

        else:
            st.error('❌ No se pudo calcular Tiempo recorrido debido a datos faltantes.')

    except Exception as e:
        st.error(f'❌ No se pudo cargar el archivo de datos. Error: {e}')
        return

    #~~~~~ Apartado Tiempo promedio ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.header('Promedios de viaje', anchor='secc_promedio_viaje')
    #switch
    opcion_promedio = st.toggle('Mostrar los promedios del tiempo de viaje', value=True)

    if opcion_promedio and datos_filtrados is not None and not datos_filtrados.empty:
        try:
            #Validar que existe "Travel_Time"
            if 'Travel_Time' not in datos_filtrados.columns:
                st.error(f'❌ La columna de "Travel_Time" no existe en los datos filtrados')
            else:
                # Si Travel_Time es de tipo Timedelta, calcular el promedio directamente
                if pd.api.types.is_timedelta64_dtype(datos_filtrados['Travel_Time']):

                    #Calcular promedio a segundos
                    promedio_segundos = datos_filtrados['Travel_Time'].dt.total_seconds().mean()

                    #Convertir el promedio de segundos a formato de hora (HH:MM:SS)
                    horas = int(promedio_segundos // 3600)
                    minutos = int((promedio_segundos % 3600) // 60)
                    segundos = int(promedio_segundos % 60)
                    promedio_formateado = f'{horas:02d}:{minutos:02d}:{segundos:02d}'

                    #resultado
                    st.write(f'El promedio del tiempo de viaje para los datos seleccionados')
                    st.write(f' Año: {year_selected}')
                    st.write(f' Mes: {month_selected}')
                    st.write(f' Promedio del Viaje es: **{promedio_formateado}**')
                else:
                    st.error('❌ La columna de Travel_Time no es de tipo TimeDelta')
        except Exception as e:
            st.error(f'❌ No se pudo calcular el promedio del tiempo de viaje. Error:{str(e)}')
    else:
        st.write('Activa el interruptor para ver el promedio del tiempo de viaje')
    # =========== FIN CONTENIDO ==================================


#----- Ejecución de la Aplicación --------------------------------
if __name__ == '__main__':
    main()


