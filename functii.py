import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype


def nan_replace(x: np.ndarray):
    is_nan = np.isnan(x)
    k = np.where(is_nan)
    x[k] = np.nanmean(x[:, k[1]], axis=0)


def nan_replace_t(t: pd.DataFrame):
    for coloana in t.columns:
        if t[coloana].isna().any():
            if is_numeric_dtype(t[coloana]):
                t.fillna({coloana: t[coloana].mean()}, inplace=True)
            else:
                t.fillna({coloana: t[coloana].mode()[0]}, inplace=True)


def acp(x: np.ndarray, std=True, nlib=0):
    n, m = x.shape
    x_ = x - np.mean(x, axis=0)
    if std:
        x_ = x_ / np.std(x, axis=0, ddof=nlib)
    r_v = (1 / (n - nlib)) * x_.T @ x_
    valp, vecp = np.linalg.eig(r_v)
    # print(valp)
    k = np.flip(np.argsort(valp))
    # print(k)
    alpha = valp[k]
    a = vecp[:, k]
    c = x_@a
    if std:
        r = a*np.sqrt(alpha)
    else:
        r = np.corrcoef(x_,c,rowvar=False)[:m,m:]
    return alpha,a,c,r

def tabelare_varianta(alpha):
    t = pd.DataFrame()
    t["Varianta"] = alpha
    t["Varianta cumulata"] = np.cumsum(alpha)
    proc = alpha*100/sum(alpha)
    t["Procent varianta"]=proc
    t["Procent cumulat"]=np.cumsum(proc)
    t.index = ["C"+str(i+1) for i in range(len(alpha))]
    t.index.name = "Componenta"
    return t

def salvare(x:np.ndarray, nume_linii=None,
            nume_coloane=None, nume_fisier=None):
    t = pd.DataFrame(x, nume_linii, nume_coloane)
    if nume_fisier is not None:
        t.to_csv(nume_fisier)
    return t

def calcul_criterii(alpha,procent_minimal=70):
    m = len(alpha)
    procent_cumulat = np.cumsum(alpha) * 100 / m
    k1 = np.where(procent_cumulat > procent_minimal)[0][0] + 1
    k2 = len(np.where(alpha > 1)[0])
    eps = alpha[:m - 1] - alpha[1:]
    sigma = eps[:m - 2] - eps[1:]
    exista_negative = sigma < 0
    if any(exista_negative):
        k3 = np.where(exista_negative)[0][0] + 2
    else:
        k3 = np.nan
    return (k1, k2, k3)