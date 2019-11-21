import matplotlib.pyplot as plt
import re
import chart_studio.plotly as py
import plotly.graph_objs as go
# from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
from math import pi

#Graficas
def print_g1(data):
    #Graficar las naciones de cada jugador
    tm = data['nation'].value_counts()
    plt_data = [go.Bar(
        x = tm.index,
        y = tm
        )]
    layout = go.Layout(
        autosize=False,
        width=500,
        height=500,
        title = "Count of players nations"
    )
    fig = go.Figure(data=plt_data, layout=layout)
    iplot(fig)

def print_g2(data):
    #Juegos por superficie
    tm = data['court_surface'].value_counts()
    plt_data = [go.Bar(
        x = tm.index,
        y = tm
        )]
    layout = go.Layout(
        autosize=False,
        width=500,
        height=500,
        title = "Games per court surface"
    )
    fig = go.Figure(data=plt_data, layout=layout)
    iplot(fig)

def print_g3(data):
    #Juegos por año
    tm = data['court_surface'].value_counts()
    plt_data = [go.Bar(
        x = tm.index,
        y = tm
        )]
    layout = go.Layout(
        autosize=False,
        width=500,
        height=500,
        title = "Games per court surface"
    )
    fig = go.Figure(data=plt_data, layout=layout)
    iplot(fig)

def print_top10_most_won(df):
    df.sort_values(by = 'games_won',ascending = False)[['player_id', 'games_won', 'year','nation']].head(10)

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
        radarPlot(df,i,categorias,my_palette(i), df[df.columns[1]][df.index[i]])
        #radarPlot(df,i,categorias,my_palette(i), df["Title"][i])