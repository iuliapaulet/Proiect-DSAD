import sys

import pandas as pd
import numpy as np
from functii import (nan_replace_t,
                     acp, tabelare_varianta, salvare)
from grafice import(scree_plot, show, corelograma, scatter)

np.set_printoptions(5, threshold=sys.maxsize, suppress=True)

set_date = pd.read_csv("date_in/date_dsad.csv", index_col=0)
#print(set_date)

var_obs = list(set_date)[0:]
#print(var_obs)
exista_nan = set_date.isna().any().any()
if exista_nan:
    nan_replace_t(set_date)

x = set_date.values
alpha, a, c, r = acp(x)
# print(alpha)

# Analiza variantei componentelor
t = tabelare_varianta(alpha)
# print(t)
t.to_csv("date_out/Varianta.csv")

k = scree_plot(alpha)
print(k)
print("Numar componente semnificative")
print("Varianta minima:",k[0])
print("Kaiser:",k[1])
print("Cattell:",k[2])
# Numar componente semnificative
k_ = max([value for value in k if value is not None])

# Analiza corelatiilor dintre variabile si componente
t_r = salvare(r,var_obs,t.index,"date_out/r.csv")
corelograma(t_r,valori=True)

# Analiza scorurilor
s = c/np.sqrt(alpha)
t_s = salvare(s,set_date.index,t_r.columns,"date_out/s.csv")
t_c = salvare(c,set_date.index,t_r.columns,"date_out/c.csv")
for i in range(2,k_+1):
    scatter(t_s,eticheta_y="C"+str(i))
    scatter(t_r,eticheta_y="C"+str(i),corelatii=True)


# Calcul metrici
# Valorile cosinus

c2 = c*c
cosinus = (c2.T/np.sum(c2,axis=1)).T
t_cosinus = salvare(cosinus,t_s.index,t_s.columns,"date_out/cosinus.csv")
t_cosinus_r = t_cosinus.apply(
    lambda x: pd.Series(x.index[np.flip(x.argsort()) ]), axis=1
)
t_cosinus_r.to_csv("date_out/cosin_r.csv")

# Contributii
contrib = c2*100/np.sum(c2,axis=0)
t_contrib = salvare(contrib,t_s.index,t_s.columns,"date_out/contrib.csv")

# Comunalitati
r2 = r*r
comm =  np.cumsum(r2,axis=1)
t_comm = salvare(comm,t_r.index,t_r.columns,"date_out/comm.csv")
corelograma(t_comm,vmin=0,cmap="Reds",titlu="Comunalitati",valori=True)

show()

