
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
st.sidebar.markdown("## MENÚ DE CONFIGURACIÓN")
st.sidebar.divider()

#----- HISTOGRAMA POR MES -----------------------------------------
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
