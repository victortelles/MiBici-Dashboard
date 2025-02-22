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

#----- Funcionalidad para crear columna de Tiempo recorrido --------
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

#===== Grafica [Type] ==== Promedio de viaje (dia * semana) ==
def graf_uso_semanal(datos_filtrados):
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
        # Mostrar tabla de conteo
        st.markdown('#### 📊 Conteo de Viajes por Día de la Semana:')
        st.dataframe(count_viajes.reset_index().rename(columns={'index': 'Day', 'Day_Week': 'Cantidad de Viajes'}))

        #Creacion y Configuracion Grafica
        plt.figure(figsize=(10,6))
        sns.barplot(x=count_viajes.index, y =count_viajes.values, palette='coolwarm')

        # conf.
        plt.title('Uso de MiBici por Dia de la Semana', fontsize=16)
        plt.xlabel('Dia de la semmana', fontsize=12)
        plt.ylabel('Cantidad de Viajes', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        #Mostrar grafico
        st.pyplot(plt)

    except Exception as e:
        st.error(f'❌ No se pudo generar la gráfica de uso semanal {e}')

#===== Grafica Histograma === Hombres vs Mujeres = Uso de MiBici =
def graf_gender_versus(datos_filtrados):
    '''Grafica para mostrar la comparativa de H vs M al usar MiBici durante la semana'''
    try:
        #Validar que los datos no esten
        if datos_filtrados is None or datos_filtrados.empty:
            st.error('❌ No hay datos filtrados para generar la grafica')
            return
        # Convertir a fecha a datetime
        datos_filtrados['Trip_Start'] = pd.to_datetime(datos_filtrados['Trip_Start'])
        #Obtener el dia de la semana (0 = Lunes - 6 = Domingo)
        datos_filtrados['Day_Week'] = datos_filtrados['Trip_Start'].dt.dayofweek
        #Diccionario para mapear el # a dias
        dias_semana = {0: 'Lunes', 1: 'Martes', 2: 'Miercoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sabado', 6: 'Domingo',}
        #Remplazar los valores numericos a nombre del dia
        datos_filtrados['Day_Week'] = datos_filtrados['Day_Week'].map(dias_semana)
        #Conteo de # viajes por genero durante la semana dia de la semana
        count_gender = datos_filtrados.groupby(['Day_Week', 'Gender']).size().reset_index(name='Cantidad_Viajes')

        # Pivotear la tabla para que cada género sea una columna
        count_pivot = count_gender.pivot(index='Day_Week', columns='Gender', values='Cantidad_Viajes').fillna(0)

        # Mostrar la tabla
        st.markdown('#### 📊 Conteo de Viajes por Día y Género:')
        st.dataframe(count_pivot)

        # Crear la gráfica
        plt.figure(figsize=(10,6))
        colores = ['#ff69b4', '#1f77b4']
        count_pivot.plot(kind='bar', stacked=False, color=colores, alpha=0.8, width=0.8)

        # Configuración del gráfico
        plt.title('Comparación de Uso de MiBici entre Hombres y Mujeres', fontsize=16)
        plt.xlabel('Día de la Semana', fontsize=12)
        plt.ylabel('Cantidad de Viajes', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Genero', labels=['Mujeres', 'Hombres'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Mostrar gráfico en Streamlit
        st.pyplot(plt)

    except Exception as e:
        st.error(f'❌ No se pudo generar la grafica')


#~~~~~~~~~~~~~~~~~~~~~~~~~ Interfaz APP ~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    #----- Configuracion inicial --------------------------------
    st.image(io.imread(LOGO_PATH), width=200)
    st.title('Datos de mi Bici (2014-2024)')
    st.subheader(':blue[MiBici]')
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
        st.markdown(f'## *Mostrando datos:*')
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
    st.markdown('### Contador de estaciones')
    st.text('Registros de cada estacion (Salida/llegada).')

    if datos_filtrados is not None and not df.empty:
        tipo_conteo = st.radio("Selecciona qué conteo deseas ver:", ["Salen", "Llegan"], horizontal=True)

        # Llamada a la función conteo_estacion con los datos de MiBici, nomenclatura y tipo de conteo
        conteo_df = conteo_estacion(datos_filtrados, nomenclatura_df, tipo_conteo)

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
    #~~~~~ Apartado mostrar edades ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    st.markdown('### Edades')
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
    st.markdown('### Graficos')
    st.image(io.imread(GRAPH_PATH), width=600)

    #st.markdown('### Gráfica de Viajes por Mes y Año')
    if datos_filtrados is not None and not datos_filtrados.empty:
        #----- Llamada a funcion de graficos ---------------------
        st.markdown('### Viajes por Mes y Año')
        graf_viaje(datos_filtrados, opcion_filtrado, year_selected, month_selected)
        st.markdown('### Uso de MiBici en durante la semana')
        graf_uso_semanal(datos_filtrados)
        st.markdown('### Comparativa Hombres vs Mujeres en Uso de MiBici durante la semana')
        graf_gender_versus(datos_filtrados)

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
    st.markdown('### Promedios de viaje')
    #Añadir un switch
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

#=================================================================
#===== Grafica Histograma ==== Distancia recorrida ===============
#=================================================================

#=================================================================
#===== Grafica Boxplot ==== Tiempo de viaje vs Ruta / genero =====
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

