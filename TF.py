import numpy as np
import seaborn as sns; sns.set()
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.plotting import parallel_coordinates
from pandas.plotting import andrews_curves
import tkinter as tk
from tkinter import font as tkfont
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
from pylab import plot, show, xlabel, ylabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.animation as animation
from matplotlib import style
import re
import chart_studio.plotly as py
import plotly.graph_objs as go
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
# My modules
from functions.preprocessing import *
from functions.graphips import *
from functions.dataset import *
from functions.datatreatment import *
from functions.algorithms import *

from mpl_toolkits.mplot3d import Axes3D # Instalar si no corre

style.use("ggplot")

class My_GUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (FrameHandler, show_dataSet, select_Data, normalizar, weightedAverageFrame, 
        MaximinFrame, MinimaxFrame, LeximinFrame, LeximaxFrame, SkylinesFrame, BestPlayersFrame,
        KMeansFrame, PCAFrame, PCAFrame2, KNNFrame, PcaSkylinesFrame, PcaKmeansFrame):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame


            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("FrameHandler")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()
    
class FrameHandler(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        label = tk.Label(self, text="ATP dataset", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        read_data = tk.Button(self, text="Leer dataset",
                            command=lambda: controller.show_frame("show_dataSet"))
        read_data.pack()

        read_data2 = tk.Button(self, text="Seleccionar data",
                            command=lambda: controller.show_frame("select_Data"))
        read_data2.pack()

        normalizar = tk.Button(self, text="normalizar data",
                            command=lambda: controller.show_frame("normalizar"))
        normalizar.pack()

        promedioPonderado = tk.Button(self, text="calcular promedio ponderado",
                            command=lambda: controller.show_frame("weightedAverageFrame"))
        promedioPonderado.pack()

        MaximinF = tk.Button(self, text="calcular Maximin",
                            command=lambda: controller.show_frame("MaximinFrame"))
        MaximinF.pack()

        MinimaxF = tk.Button(self, text="calcular Minimax",
                            command=lambda: controller.show_frame("MinimaxFrame"))
        MinimaxF.pack()   

        LeximinF = tk.Button(self, text="calcular Leximin",
                            command=lambda: controller.show_frame("LeximinFrame"))
        LeximinF.pack()  

        LeximaxF = tk.Button(self, text="calcular Leximax",
                            command=lambda: controller.show_frame("LeximaxFrame"))
        LeximaxF.pack()       

        SkylinesF = tk.Button(self, text="calcular Skylines",
                            command=lambda: controller.show_frame("SkylinesFrame"))
        SkylinesF.pack()  

        BestPlayersF = tk.Button(self, text="mostrar mejores jugadores",
                            command=lambda: controller.show_frame("BestPlayersFrame"))
        BestPlayersF.pack()  

        KmeansF = tk.Button(self, text="mostrar agrupamiento kmeans",
                            command=lambda: controller.show_frame("KMeansFrame"))
        KmeansF.pack() 

        PCAF = tk.Button(self, text="reducir dimensionalidad",
                            command=lambda: controller.show_frame("PCAFrame"))
        PCAF.pack()

        PCAF2 = tk.Button(self, text="reducir dimensionalidad 2",
                            command=lambda: controller.show_frame("PCAFrame2"))
        PCAF2.pack()  

        KNN = tk.Button(self, text="Classificacion de datos (KNN)",
                            command=lambda: controller.show_frame("KNNFrame"))
        KNN.pack() 

        PcaSkylines = tk.Button(self, text="Combinacion de metodos y/o Agrupamiento",
                            command=lambda: controller.show_frame("PcaSkylinesFrame"))
        PcaSkylines.pack() 

        PcaKmeans = tk.Button(self, text="Combinacion de PCA & K-Means",
                            command=lambda: controller.show_frame("PcaKmeansFrame"))
        PcaKmeans.pack() 

        graph_nationsF = tk.Button(self, text="ver grafica 1",
                            command=graph_nations)
        graph_nationsF.pack()  

        graph_surfaceF = tk.Button(self, text="ver grafica 2",
                            command=graph_surface)
        graph_surfaceF.pack()  

        graph_yearF = tk.Button(self, text="ver grafica 3",
                            command=graph_years)
        graph_yearF.pack()  
        
#Lectura de datos
class show_dataSet(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text = tk.Text(self, width=1, height=1)

        text.insert(tk.END, str(data.head()))

        text.pack(fill="both", expand=True)

        label = tk.Label(self, text="Lectura de dataset", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()

class select_Data(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text = tk.Text(self, width=1, height=1)

        data = select_data(data)

        text.insert(tk.END, str(data.head()))
        text.pack(fill="both", expand=True)

        label = tk.Label(self, text="Data seleccionada", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()

# Pre-procesamiento
class normalizar(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text = tk.Text(self, width=1, height=1)

        drop_na_values(data)
        indices = [1,6,7,8,9]
        data = normalize_columns(indices,data)

        text.insert(tk.END, str(data.head()))
        text.pack(fill="both", expand=True)

        label = tk.Label(self, text="Data normalizada", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()

#Tratamiendo de datos
class weightedAverageFrame(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text = tk.Text(self, width=1, height=1)

        indices_wa = [6,7,8,9]
        pesos = [1,2,3,3]
        dfWA = weightedAverage(data, pesos, indices_wa)

        text.insert(tk.END, str(dfWA.head()))
        text.pack(fill="both", expand=True)

        label = tk.Label(self, text="Promedio ponderado", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        skyline = tk.Button(self, text="ver skylines",
                           command= lambda: radarAllPlot(dfWA.head(4), indices_wa))
        skyline.pack()

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()

class MaximinFrame(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text = tk.Text(self, width=1, height=1)

        indices = [7,8,9]
        dfMM = maximin(data, indices)

        text.insert(tk.END, str(dfMM.head()))
        text.pack(fill="both", expand=True)

        label = tk.Label(self, text="Maximin", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()

class MinimaxFrame(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text = tk.Text(self, width=1, height=1)

        indices_maximin = [7,8,9]
        dfMM = maximin(data, indices_maximin)

        text.insert(tk.END, str(dfMM.head()))
        text.pack(fill="both", expand=True)

        label = tk.Label(self, text="Minimax", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()

class LeximinFrame(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text = tk.Text(self, width=1, height=1)

        indices = [7,8,9]
        dfLexmin = leximin(data, indices)

        text.insert(tk.END, str(dfLexmin.head()))
        text.pack(fill="both", expand=True)

        label = tk.Label(self, text="Leximin", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()

class LeximaxFrame(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text = tk.Text(self, width=1, height=1)

        ## Leximax
        indices = [7,8,9]
        dfLeximax = leximax(data, indices)

        text.insert(tk.END, str(dfLeximax.head()))
        text.pack(fill="both", expand=True)

        label = tk.Label(self, text="Leximax", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()

class SkylinesFrame(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text = tk.Text(self, width=1, height=1)

        indices = [7,8,9]
        dfSky = skylines(data, indices)

        text.insert(tk.END, str(dfSky.head()))
        text.pack(fill="both", expand=True)

        label = tk.Label(self, text="Skylines", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        skyline = tk.Button(self, text="ver skylines",
                           command= lambda: radarAllPlot(dfSky.head(4), indices))
        skyline.pack()

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()

class BestPlayersFrame(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text = tk.Text(self, width=1, height=1)

        bestPlayersDF = data.sort_values(by = 'games_won',ascending = False)[['player_id', 'games_won', 'year','nation']].head(5)

        text.insert(tk.END, str(bestPlayersDF))
        text.pack(fill="both", expand=True)

        label = tk.Label(self, text="Mejores 5 jugadores", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        verGraficas = tk.Button(self, text="ver Graficas",
                           command=lambda: bestPlayers(bestPlayersDF))
        verGraficas.pack()

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()

class KMeansFrame(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        # textentry = tk.Entry(self, width=20, bg="white")

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text = tk.Text(self, width=1, height=1)

        label = tk.Label(self, text="Agrupamiento en cluster (Kmeans) \n grupos: 'prize_money', 'num_sets', 'sets_won', 'games_won', 'games_against'", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        # kmeans = tk.Button(self, text="iniciar agrupamiento",
        #                    command=lambda: kmeansGrouping(data, textentry.get(), text))
        # kmeans.pack()

        # Agrupamiento de datos K-means 
        dataP = data[['prize_money', 'num_sets', 'sets_won', 'games_won', 'games_against']]
        X = np.array(dataP)
        kmeansDF = kmeans(X, data, 5, 3)

        text.insert(tk.END, str(kmeansDF))

        text.pack(fill="both", expand=True)

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()

class PCAFrame(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        # textentry = tk.Entry(self, width=20, bg="white")

        label = tk.Label(self, text="Redución de dimensionalidad (PCA)", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text1 = tk.Text(self, width=20, height=3)
        text2 = tk.Text(self, width=20, height=3)

        dataP = data[['prize_money', 'num_sets', 'sets_won', 'games_won', 'games_against']]
        X = np.array(dataP)

        pca = PCA(n_components=2)
        pca.fit(X)

        text1.insert(tk.END, str(pca.n_components_))
        text1.pack()

        text2.insert(tk.END, str(pca.explained_variance_))
        text2.pack()

        scaterPlot = tk.Button(self, text="Ver gráfica de disperisón",
                           command=lambda: scartterPlot(X))
        scaterPlot.pack()

        pca = PCA(n_components=1)
        pca.fit(X)
        X_pca = pca.transform(X)
        X_new = pca.inverse_transform(X_pca)

        scaterPlot2 = tk.Button(self, text="Ver gráfica de disperisón conjunta",
                           command=lambda: scartterPlot2(X, X_new))
        scaterPlot2.pack()

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()

class PCAFrame2(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        # textentry = tk.Entry(self, width=20, bg="white")

        label = tk.Label(self, text="Redución de dimensionalidad 2 (PCA)", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        text1 = tk.Text(self, width=300, height=10)
        text2 = tk.Text(self, width=300, height=10)

        dataP = data[['prize_money', 'num_sets', 'sets_won', 'games_won', 'games_against']]
        X = np.array(dataP)

        pca = PCA().fit(X)

        pca2 = PCA(n_components=2)
        principalComponents = pca2.fit_transform(X)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = ['principal component 1', 'principal component 2'])

        finalDf = pd.concat([principalDf, data[['court_surface']]], axis = 1)
        
        text1.insert(tk.END, str(principalDf.head()))
        text1.pack()

        text2.insert(tk.END, str(finalDf.head()))
        text2.pack()

        PlotVariance = tk.Button(self, text="Ver gráfica de varianza",
                           command=lambda: variancePlot(pca))
        PlotVariance.pack()

        PlotPCA = tk.Button(self, text="Ver gráfica de PCA",
                           command=lambda: PCAPlot(finalDf, pca2))
        PlotPCA.pack()

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()


class KNNFrame(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        label = tk.Label(self, text="KNN", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        global data
        feature_columns = ['prize_money', 'num_sets', 'sets_won','games_won','games_against']
        X = data[feature_columns].values
        y = data['nation'].values

        le = LabelEncoder()
        y = le.fit_transform(y)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        data_for_KNN = data.head(400)

        KNNplot = tk.Button(self, text="Ver gráfica 1 de clasificacion KNN",
                           command=lambda: KNNPlot1(X,data_for_KNN))
        KNNplot.pack()

        KNNplot = tk.Button(self, text="Ver gráfica 2 de clasificacion KNN",
                           command=lambda: KNNPlot2(X,data_for_KNN))
        KNNplot.pack()

        KNNplot = tk.Button(self, text="Ver gráfica 3 de clasificacion KNN",
                           command=lambda: KNNPlot3(X,y,data_for_KNN))
        KNNplot.pack()

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()


class PcaSkylinesFrame(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        label = tk.Label(self, text="PCA & Skylines", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        global data
        text = tk.Text(self, width=1, height=1)

        dataP = data[['prize_money', 'num_sets', 'sets_won', 'games_won', 'games_against']]
        X = np.array(dataP)
        
        pca2 = PCA(n_components=3)
        principalComponents = pca2.fit_transform(X)
        principalDf = pd.DataFrame(data = principalComponents
                    , columns = ['principal component 1', 'principal component 2','principal component 3'])

        finalDf = pd.concat([principalDf, data[['court_surface']]], axis = 1)

        indices = [0,1,2]
        dfSky = skylines(finalDf, indices)

        text.insert(tk.END, str(dfSky.head()))
        text.pack(fill="both", expand=True)

        label = tk.Label(self, text="Skylines", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        skyline = tk.Button(self, text="ver skylines",
                           command= lambda: radarAllPlot(dfSky.head(4), indices))
        skyline.pack()

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()


class PcaKmeansFrame(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.controller = controller

        label = tk.Label(self, text="PCA & Skylines", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        global data
        text = tk.Text(self, width=1, height=1)

        dataP = data[['prize_money', 'num_sets', 'sets_won', 'games_won', 'games_against']]
        X = np.array(dataP)
        kmeansDF = kmeans(X, data, 5, 3)

        text.insert(tk.END, str(kmeansDF))

        text.pack(fill="both", expand=True)

        button = tk.Button(self, text="Regresar",
                           command=lambda: controller.show_frame("FrameHandler"))
        button.pack()


#Mostrar graficas
def graph_nations():
    global data
    tm = data['nation'].value_counts()
    plt.bar(tm.index, tm) #funciona, pero no muy bien
    plt.xticks(rotation=90)
    plt.title('Número de jugadores por país')
    plt.show()

def graph_surface():
    global data
    tm = data['court_surface'].value_counts()
    plt.bar(tm.index, tm) 
    plt.xticks(rotation=90)
    plt.title('Cantidad de juegos por superficie de la cancha')
    plt.show()

def graph_years():
    global data
    tm = data['year'].value_counts()
    plt.bar(tm.index, tm) 
    plt.xticks(rotation=90)
    plt.title('Juegos por año')
    plt.show()

def bestPlayers(df):
    labels = df['player_id'].values[0], df['player_id'].values[1], df['player_id'].values[2], df['player_id'].values[3], df['player_id'].values[4]
    total = df['games_won'].values[0] + df['games_won'].values[1] + df['games_won'].values[2] + df['games_won'].values[3] + df['games_won'].values[4]
    percentage = [(df['games_won'].values[0]*100)/total, (df['games_won'].values[1]*100)/total, (df['games_won'].values[2]*100)/total, (df['games_won'].values[3]*100)/total, (df['games_won'].values[4]*100)/total]
    sizes = percentage

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

def radarPlot(df, row, categorias, color,title):
    N = len(categorias)
    #repetir el primer valor para tener una figura cerrada (poligono)
    valores = df.loc[df.index[row]].values[categorias].flatten().tolist()    
    valores += valores[:1]
    #calcular el angulo
    angulos = [n / float(N) * 2 * pi for n in range(N)]
    angulos += angulos[:1]
    #inicializar el plot
    ax = plt.subplot(3, 2, row + 1, polar=True, )
    # primer eje arriba:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    #un eje por atributo + etiquetas
    etiquetas = [df.columns[i] for i in categorias]
    plt.xticks(angulos[:-1], etiquetas, color='grey', size=8)
    ax.set_rlabel_position(0)
    #dibujar ticks de los ejes
    tic = 5
    plt.yticks([i * (1.0 / tic) for i in range(1,tic)], [str(i * (1.0 / tic)) for i in range(1,tic)], color="grey", size=7)
    plt.ylim(0,1)
    #plotear
    ax.plot(angulos, valores, color=color, linewidth=2, linestyle='solid')
    ax.fill(angulos, valores, color=color, alpha=0.4)
    plt.title(title, size=11, color=color, y=1.1)

def radarAllPlot(df,categorias):
    my_dpi=96
    plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df.index))
    for i in range(len(df.index)):
        #print(df.columns[1])
        #print()
        radarPlot(df,i,categorias,my_palette(i), df[df.columns[3]][df.index[i]])
        #radarPlot(df,i,categorias,my_palette(i), df["Title"][i])
    plt.show()

def scartterPlot(X):
    plt.title("Gráfica de dispersión", size=11, y=1.1)
    plt.scatter(X[:, 0], X[:, 1])
    plt.axis('equal')
    plt.show()


def scartterPlot2(X, X_new):
    plt.title("Gráfica de dispersión conjunta", size=11, y=1.1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.show()

def variancePlot(pca):
    plt.title("Gráfica de varianza")
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('numero de componentes')
    plt.ylabel('varianza explicada acumulada')
    plt.show()

def PCAPlot(finalDf, pca2):
    plt.title("Gráfica de relación de componentes")
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['Clay', 'Hard', 'Grass', 'Carpet']
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['court_surface'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()

def KNNPlot1(X,data):
    plt.figure(figsize=(15,10))
    parallel_coordinates(data.drop(["court_surface","year","player_id","opponent_id","tournament"], axis=1), "nation")
    plt.title('Coordenadas Paralelas', fontsize=20, fontweight='bold')
    plt.xlabel('Caracteristicas', fontsize=15)
    plt.ylabel('Valores de las Caracteristicas', fontsize=15)
    plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
    plt.show()
def KNNPlot2(X,data):
    plt.figure(figsize=(15,10))
    andrews_curves(data.drop(["court_surface","year","player_id","opponent_id","tournament"], axis=1), "nation")
    plt.title('Andrews Curves Plot', fontsize=20, fontweight='bold')
    plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
    plt.show()
def KNNPlot3(X,y,data):
    fig = plt.figure(1, figsize=(20, 15))
    ax = Axes3D(fig, elev=48, azim=134)
    ax.scatter(X[:, 2], X[:, 3], X[:, 4], c=y,
            cmap=plt.cm.Set1, edgecolor='k', s = X[:, 3]*50)

    for name, label in [('Virginica', 0), ('Setosa', 1), ('Versicolour', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                X[y == label, 1].mean(),
                X[y == label, 2].mean(), name,
                horizontalalignment='center',
                bbox=dict(alpha=.5, edgecolor='w', facecolor='w'),size=25)

    ax.set_title("3D visualization", fontsize=40)
    ax.set_xlabel("num_sets", fontsize=25)
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("sets_won", fontsize=25)
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("games_won", fontsize=25)
    ax.w_zaxis.set_ticklabels([])
    plt.show()

# Algoritmos (por arreglar)
def kmeansGrouping(df, seleccion, text):
    dataP = df[['prize_money', 'num_sets', 'sets_won', 'games_won', 'games_against']]
    X = np.array(dataP)
    kmeansDF = kmeans(X, df, 5, seleccion)
    text.insert(tk.END, str(kmeansDF))

# class graphs(tk.Frame):

#     def __init__(self, parent, controller):
#         tk.Frame.__init__(self, parent)

#         self.controller = controller

#         label = tk.Label(self, text="Número de jugadores por país", font=controller.title_font)
#         label.pack(side="top", fill="x", pady=10)

#         global data # quick and dirty way to access `df`, think about making it an attribute or creating a function that returns it
        
#         # f = Figure(figsize=(5,5), dpi=100)
#         # a = f.add_subplot(111)

#         # xList = []
#         # yList = []

#         # for eachLine in data:
#         #     if len(eachLine) > 1:
#         #         x, y = eachLine.split(',')
#         #         xList.append(int(x))
#         #         yList.append(int(y))
        
#         # a.clear()

#         tm = data['nation'].value_counts()

#         # a.plot(tm.index, tm)
#         # a.set_title("Número de jugadores por país")

#         # for tick in a[0].get_xticklabels():
#         #     tick.set_rotation(90)
#         # f.align_xlabels()

#         # canvas = FigureCanvasTkAgg(f, self)
#         # canvas.draw()
#         # canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)


#         plt.bar(tm.index, tm) #funciona, pero no muy bien
#         plt.xticks(rotation=90)
#         plt.show(self)

#         # fig = Figure(figsize=(5, 4), dpi=100)

#         # fig.add_subplot(111).plot()

#         # tm = data['nation'].value_counts()
#         # plt_data = [go.Bar(
#         #     x = tm.index,
#         #     y = tm
#         #     )]
#         # layout = go.Layout(
#         #     autosize=False,
#         #     width=1000,
#         #     height=500,
#         #     title = "Count of players nations"
#         # )
        
#         # canvas = FigureCanvasTkAgg(fig, self)
#         # canvas.draw()
#         # canvas.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand = True)

#         # toolbar = NavigationToolbar2Tk(canvas, root)
#         # toolbar.update()
#         # canvas.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand = True)

#         button = tk.Button(self, text="Regresar",
#                            command=lambda: controller.show_frame("FrameHandler"))
#         button.pack()

#Data is global 
data = read_dataset("./resources/all_matches.csv",pd)

if __name__ == "__main__":
    app = My_GUI()
    app.state('zoomed')
    app.mainloop()


# if __name__ == "__main__":
    # #Read dataset from file
    # data = read_dataset("./resources/all_matches.csv",pd)

#     #Select data we need
#     data = select_data(data)

#     # Pre-procesamiento
#     ## Drop na values
#     drop_na_values(data)
#     ## Normalizate columns
#     indices = [1,6,7,8,9]
#     data = normalize_columns(indices,data)

#     #Tratamiendo de datos

#     ## Promedio ponderado
#     indices_wa = [6,7,8,9]
#     pesos = [1,2,3,3]
#     dfWA = weightedAverage(data, pesos, indices_wa)
#     #print_details(dfWA)

#     ## Maximin
#     indices_maximin = [7,8,9]
#     dfMM = maximin(data, indices_maximin)
#     #print_details(dfMM)
    
#     ## Maximax
#     indices_minimax = [7,8,9]
#     dfMinmax = minimax(data, indices_minimax)
#     #print_details(dfMinmax)

#     ## Leximin
#     indices_leximin = [7,8,9]
#     dfLexmin = leximin(data, indices_leximin)
#     #print_details(dfLexmin)

#     ## Leximax
#     indices_leximax = [7,8,9]
#     dfLeximax = leximax(data, indices_leximax)
#     print_details(dfLeximax)

#     ##Skylines
#     indices_skylines = [7,8,9]
#     dfSky = skylines(data, indices_skylines)
#     #print_details(dfSky)

#     #Skylines graphips
#     radarAllPlot(dfWA.head(4),indices_skylines)
#     radarAllPlot(dfMM.head(4),indices_skylines)
#     radarAllPlot(dfMinmax.head(4),indices_skylines)
#     radarAllPlot(dfLexmin.head(4),indices_skylines)
#     radarAllPlot(dfLeximax.head(4),indices_skylines)
#     radarAllPlot(dfSky.head(4),indices_skylines)

#     # Agrupamiento de datos K-means 
#     dataP = data[['prize_money', 'num_sets', 'sets_won', 'games_won', 'games_against']]
#     dataP.head()
#     X = np.array(dataP)
#     kmeans(X, data, 5, 3)

#     # Reduccion de dimensionalidad con PCA
#     # pca = PCA().fit(X)
#     # plt.plot(np.cumsum(pca.explained_variance_ratio_))
#     # plt.xlabel('numero de componentes')
#     # plt.ylabel('varianza explicada acumulada')