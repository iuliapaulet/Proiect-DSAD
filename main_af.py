import pandas as pd
import numpy as np
from functii import *
from grafice import *
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo

np.set_printoptions(precision=5, threshold=10000, suppress=True)

# Citirea datelor
set_date = pd.read_csv("date_in/date_dsad.csv", index_col=0)
var_obs = list(set_date.columns)  # Variabilele observate

# Transformarea datelor pentru analiză
x = set_date[var_obs].values

# Test Bartlett pentru factorabilitate
test_bartlett = calculate_bartlett_sphericity(x)
print("Test Bartlett:", test_bartlett)
if test_bartlett[1] > 0.001:
    print("Nu există factori comuni! Testul Bartlett a eșuat.")
    exit(0)

# Calculul KMO
kmo = calculate_kmo(x)
print("KMO Total:", kmo[1])
t_kmo = pd.DataFrame(
    {"KMO": np.append(kmo[0], kmo[1])},
    index=var_obs + ["KMO Total"]
)
t_kmo.to_csv("date_out/kmo.csv")

# Plot corelogramă KMO
try:
    corelograma(t_kmo.iloc[:-1], vmin=0, vmax=1, cmap="Blues", titlu="Index KMO", valori=True)
except ValueError as e:
    print("Eroare în generarea corelogramei KMO:", e)

# Construire model de analiză factorială
# fara rotatie
n, m = x.shape
model_af_none = FactorAnalyzer(m, rotation=None)
model_af_none.fit(x)


# Analiza variantei + plot varianta
varianta = model_af_none.get_factor_variance()
print(varianta)
etichete_factori = ["F" + str(i) for i in range(1, m + 1)]
tabel_varianta = pd.DataFrame(
    {
        "Varianta": varianta[0],
        "Procent varianta": varianta[1] * 100,
        "Procent cumulat": varianta[2] * 100
    }, etichete_factori
)
tabel_varianta.to_csv("date_out/varianta_af_fara_rotatie.csv")
alpha = varianta[0]
criterii = calcul_criterii(alpha)
#plot_varianta(alpha, criterii, procent_minimal=70, eticheta_x="Factor")
show()

# Numar factori semnificativi
# k = np.nanmin(criterii)
k = int(np.nanmean(criterii))

# Analiza corelatiilor variabile-factori
l = model_af_none.loadings_
t_l = pd.DataFrame(l, var_obs, etichete_factori)
t_l.to_csv("date_out/corelatii_factoriale_fara_rotatie.csv")
corelograma(t_l)
for j in range(2, k + 1):
    scatter(t_l, "F1", "F" + str(j), "Plot corelatii factoriale", corelatii=True)
show()

# Calculul scorurilor
f = model_af_none.transform(x)
t_f = pd.DataFrame(f, set_date.index, etichete_factori)
t_f.to_csv("date_out/scoruri_fara_rotatie.csv")
for j in range(2, k + 1):
    scatter(t_f, "F1", "F" + str(j))

# cu rotatie
model_af = FactorAnalyzer(m, rotation="varimax")
model_af.fit(x)

# Analiza variantei + plot varianta
varianta = model_af.get_factor_variance()
# print(varianta)
etichete_factori = ["F" + str(i) for i in range(1, m + 1)]
tabel_varianta = pd.DataFrame(
    {
        "Varianta": varianta[0],
        "Procent varianta": varianta[1] * 100,
        "Procent cumulat": varianta[2] * 100
    }, etichete_factori
)
tabel_varianta.to_csv("date_out/varianta_factori.csv")
alpha = varianta[0]
criterii = calcul_criterii(alpha)
#plot_varianta(alpha, criterii, procent_minimal=70, eticheta_x="Factor")
show()

# Numar factori semnificativi
# k = np.nanmin(criterii)
k = int(np.nanmean(criterii))

# Analiza corelatiilor variabile-factori
l = model_af.loadings_
t_l = pd.DataFrame(l, var_obs, etichete_factori)
t_l.to_csv("date_out/corelatii_factoriale_rotatie.csv")
corelograma(t_l)
for j in range(2, k + 1):
    scatter(t_l, "F1", "F" + str(j), "Plot corelatii factoriale cu rotatie", corelatii=True)

# Calculul scorurilor
f = model_af.transform(x)
t_f = pd.DataFrame(f, set_date.index, etichete_factori)
t_f.to_csv("date_out/scoruri_factoriale_rotatie.csv")
for j in range(2, k + 1):
    scatter(t_f, "F1", "F" + str(j))

# Analiza comunalitatii si a variantei specifice
h = model_af.get_communalities()
psi = model_af.get_uniquenesses()
t_varianta_extrasa = pd.DataFrame(
    {
        "Comunalitate":h,
        "Varianta specifica":psi
    }, var_obs
)
t_varianta_extrasa.to_csv("date_out/comunalitati_factoriale.csv")
# Apel funcție cu vmax calculat automat
corelograma(t_varianta_extrasa, vmin=0, vmax=None, cmap="Reds", titlu="Varianta extrasa", valori=True)


# Afișarea graficelor
show()
