import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap
from geopandas import GeoDataFrame


def scree_plot(alpha, varianta_explicata=80,titlu="Plot varianta componente",x_label="Componenta"):
    fig = plt.figure(figsize=(8, 5))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu,fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel(x_label)
    ax.set_ylabel("Varianta")
    m = len(alpha)
    x = np.arange(1, m + 1)
    ax.set_xticks(x)
    ax.plot(x, alpha)
    ax.scatter(x, alpha, c="r", alpha=0.5)
    proc_cum = np.cumsum(alpha * 100 / sum(alpha))
    k1 = np.where(proc_cum > varianta_explicata)[0][0] + 1
    # print(k1)
    ax.axvline(k1, c="g", label="Varianta explicata >" + str(varianta_explicata) + "%")
    k2 = np.where(alpha > 1)[0][-1] + 1
    ax.axvline(k2, c="c", label="Kaiser")
    eps = alpha[:m-1] - alpha[1:]
    sigma = eps[:m-2] - eps[1:]
    # print(sigma)
    if any(sigma<0):
        k3 = np.where(sigma<0)[0][0] + 2
        ax.axhline(alpha[k3-1], c="m", label="Cattell")
    else:
        k3= None
    ax.legend()
    return k1,k2,k3

def show():
    plt.show()

def corelograma(t:pd.DataFrame,vmin=-1,vmax=1,
                cmap="RdYlBu",
                titlu="Corelatii variabile-componente",
                valori=False):
    # Verifică dacă toate valorile din tabel sunt numerice
    if not np.issubdtype(t.values.dtype, np.number):
        raise ValueError("Tabelul trebuie să conțină doar valori numerice.")

    # Setează vmax dacă nu este specificat
    if vmax is None:
        vmax = t.max().max()
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu,fontdict={"fontsize": 16, "color": "b"})
    heatmap(t,vmin=vmin,vmax=vmax,cmap=cmap,ax=ax,annot=valori)

def scatter(t:pd.DataFrame,
            eticheta_x="C1",
            eticheta_y="C2",
            etichete=True,
            titlu="Plot scoruri",
            corelatii=False):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    ax.set_title(titlu,fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel(eticheta_x)
    ax.set_ylabel(eticheta_y)
    ax.scatter(t[eticheta_x],t[eticheta_y],c="r")
    if corelatii:
        theta = np.arange(0,2*np.pi,0.01)
        ax.plot(np.sin(theta),np.cos(theta))
        ax.plot(0.7*np.sin(theta),0.7*np.cos(theta),c='g')
    ax.axhline(0,c="k")
    ax.axvline(0,c="k")
    if etichete:
        n= len(t)
        for i in range(n):
            ax.text(t[eticheta_x].iloc[i],
                    t[eticheta_y].iloc[i],
                    t.index[i])

def plot_harta(gdf:GeoDataFrame,t:pd.DataFrame,camp_legatura_gdf,camp_harta,titlu,cmap="RdYlBu"):
    gdf_ = gdf.merge(t,left_on=camp_legatura_gdf,right_index=True)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu,fontdict={"fontsize":16,"color":"b"})
    gdf_.plot(column=camp_harta,legend=True,cmap=cmap,ax=ax)



def plot_varianta(alpha, criterii, procent_minimal=80,eticheta_x="Componenta"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title("Plot varianta", fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel(eticheta_x)
    ax.set_ylabel("Varianta")
    x = np.arange(1, len(alpha) + 1)
    ax.set_xticks(x)
    ax.plot(x, alpha)
    ax.scatter(x, alpha, c="r", alpha=0.5)
    ax.axhline(alpha[criterii[0] - 1], c="m", label="Varianta minimala:" + str(procent_minimal) + "%")
    if criterii[0] is not None:
        ax.axhline(alpha[criterii[0] - 1], c="m", label="Varianta minimala: " + str(procent_minimal) + "%")
    if criterii[1] is not None:
        ax.axhline(1, c="c", label="Kaiser")
    if criterii[2] is not None:
        ax.axhline(alpha[criterii[2] - 1], c="g", label="Cattell")
    ax.legend()
    plt.savefig("Plot_varianta")

