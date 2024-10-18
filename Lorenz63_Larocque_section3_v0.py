#=================================
#       IMPORTS
#=================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy

#==================================================================
#       PARTIE 3 - CLIMAT DU SYSTÈME CHAOTIQUE
#==================================================================
#---------------------------------
#       CONSTANTES
#---------------------------------

DT = 0.001 # Pas de temps delta_t = 0.001 [s]
N = int(1E5) # longueur N de la simulation, en pas de temps
S = 10      # Constante Sigma
B = 2.667   # Constante Beta
R = 28.0  # Constante Rho, solution chaotique
E = 0.01 # Constante epsilon inchangée pour la partie 3

#---------------------------------------------
#       FONCTIONS DES ÉQUATIONS DE BASE
#---------------------------------------------

# Fonction Eta
# PARAMS : beta (b), rho (r), applicables selon le cas
# RETURN : eta, en fonction de l'équation du devoir, partie 1
def Eta(b,r):
    return math.sqrt(b*(r-1))

# Fonction CI
# PARAMS : eta (e), rho (r), applicables selon le cas
# RETURN : une liste avec x_t0, y_t0, z_t0, les conditions initiales au temps 0 pour x, y, z.
def CI(e, r):
    return [e, e+3, r-1]

# Fonction CI_p
# PARAMS : eta (e), rho (r), le facteur d'amplitude de perturbation (fa),
#          applicables selon le cas
# RETURN : une liste avec x_t0, y_t0, z_t0, les conditions initiales au temps 0 pour x, y, z.
def CI_p(e, r):
    global E
    return [e+E, e+3, r-1]

# Fonction CI_xyp
# PARAMS : eta (e), rho (r), la perturbation en x (xp) et en y (yp),le facteur d'amplitude de perturbation (fa),
#          applicables selon le cas
# RETURN : une liste avec x_t0, y_t0, z_t0, les conditions initiales au temps 0 pour x, y, z.
def CI_xyp(e, r, xp, yp, fa):
    epsilonP = Epsilon(fa) if fa > 0 else 0 # Modification du paramètre de perturbation epsilonP
    return [e+xp+epsilonP, e+yp, r-1]

# Fonction X_t
# PARAMS : sigma (s), x_t (x), y_t (y),
#           les constantes sigma, dt, et valeurs x et y au pas de temps actuel
# RETURNS : x(t+1), x au prochain pas de temps
def X_t(s,x,y):
    return (s*DT*(y-x))+x

# Fonction Y_t
# PARAMS : rho (r), x_t (x), y_t (y), z_t (z),
#          les constantes rho, dt, et valeurs x, y et z au pas de temps actuel
# RETURNS : y(t+1), y au prochain pas de temps
def Y_t(r,x,y,z):
    return (((x*(r-z))-y)*DT)+y

#Fonction Z_t
# PARAMS : beta (b), x_t (x), y_t (y), z_t (z),
#          les constantes beta, dt, et valeurs x, y et z au pas de temps actuel
# RETURNS : z(t+1), z au prochain pas de temps
def Z_t(b,x,y,z):
    return (((y*x)-(b*z))*DT)+z

# Fonction timeStep
# PARAMS : beta (b), rho (r), sigma (s),
#          t_array un array avec : x_t (x), y_t (y), z_t (z),
#          les constantes beta, sigma, rho, dt et les valeurs x,y,z au temps actuel
# RETURNS : une liste avec x_(t+1), y_(t+1), z_(t+1), les valeurs x,y,z au prochain pas de temps)
def timeStep(b,r,s,t_array):
    x = t_array[0][0]
    y = t_array[0][1]
    z = t_array[0][2]
    return [ X_t(s,x,y),
             Y_t(r,x,y,z),
             Z_t(b,x,y,z) ]

#---------------------------------------------
#       FONCTIONS DE CALCULS DES VALEURS
#       POUR SOLUTIONS 3.n
#---------------------------------------------
# FONCTION XYZ_array
# PARAMS : Le paramètre rho (R_n), seul paramètre changeant entre les 3 solutions
#          Les conditions initiales (CI) données par la fonction de CI définie et passée en paramètres
#          Le nombre de pas de temps (Nlen) pour une longueur N du tableau de données
# RETURNS : le array de format (N x 3) qui contient les valeurs de x, y, z à chaque pas de temps
def XYZ_array(R_n, CI,Nlen):
    # values_array : Un array initial auquel on ajoutera les valeurs au temps 0 et aux temps subséquent.
    # vArray = np.array([[0,0,0]])

    #t0_array : Valeurs de x(t0), y(t0), z(t0), selon les conditions initiales (CI)
    t0_array = np.array( [CI] )

    #Boucle qui ajoute à values_array les valeurs x, y, z, d'abord au temps 1, puis aux temps subséquents, jusqu'à 2E4 pas de temps
    for n in range(0,Nlen-1):
        if n == 0 :
            # Cas traité séparément au premier pas de temps, pour une gestion plus lisible des variables
            # tn_array est le tableau de N lignes par 3 colonnes
            # time_array est la prochaine ligne à ajouter au array de données tn_array
            time_array = np.array([timeStep(B,R_n,S,t0_array)])
            tn_array = np.concatenate([t0_array,time_array])

        else :
            # Traitement de tous les pas de temps suivants, où tn_array est toujours modifié pour contenir les valeurs du pas de temps suivants,
            # En utilisant la fonction timeStep
            time_array = np.array([timeStep(B, R_n, S, time_array)])
            tn_array = np.concatenate([tn_array, time_array], axis=0)
    return tn_array

#---------------------------------------------
#       3.1 - CLIMAT DU SYSTÈME CHAOTIQUE
#          -> ARRAYS
#---------------------------------------------
values_array = XYZ_array(
        R,                    # Argument R_n de XYZ_array()
        CI(Eta(B, R), R),   # Argument CI de XYZ_array(), avec fonction CI contrôle
        N                       # Argument Nlen de XYZ_array()
    )
perturb_array = XYZ_array(
        R,                    # Argument R_n de XYZ_array()
        CI_p(e=Eta(B,R), r=R),   # Argument CI de XYZ_array(), avec fonction CI_p perturbation, epsilon = 0.01
        N                       # Argument Nlen de XYZ_array()
    )
#Identification des minimums et des maximums pour uniformiser les histogrammes
#print(perturb_array.max(),perturb_array.min(),values_array.max(), values_array.min())

#---------------------------------------------
#       3.1 - CLIMAT DU SYSTÈME CHAOTIQUE
#          -> HISTOGRAMMES
#---------------------------------------------
# PARAMS : Varray (numpy array) - un array donné avec les valeurs contrôle
#          Parray (numpy array) - un array donné avec les valeurs perturbées
#          xi_pos (int) - la valeur de la variable xi=[x,y,z], soit 0, 1 ou 2.
# RETURNS : aucun (graphique créé et enregistré par la fonction)
def xi_hist(Varray, Parray, xi_pos):
    # Liste des variables, pour automatiser les textes des figures
    xi=["x","y","z"][xi_pos]
    # Pour chaque variable,
    # On utilisera le minimum le plus petit entre les 2 array, et le maximum le plus grand entre les deux arrays,
    # pour choisir nos limites
    bin_min=min([int(Varray[:,xi_pos].min()),
             int(Parray[:,xi_pos].min())]
            )
    bin_max=max([int(Varray[:,xi_pos].max()),
             int(Parray[:,xi_pos].max())]
            )
    #On peut maintenant choisir nos limites de bins pour les histogrammes
    bin_edges=np.linspace(bin_min, bin_max, 50)

    #Params de figure
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    fig.suptitle("3.1 - Exploration visuelle des différences", font="Avenir", size=12)


    # Params de l'hist 1
    ax1.set_title(f"Répartition des valeurs\npour la variable {xi},\nconditions initiales contrôle", font="Avenir", size=14)
    ax1.set_xlabel(f"Valeurs de {xi}", font="Avenir", size=14)
    ax1.set_ylabel(f"Nombre de valeurs de {xi}", font="Avenir", size=14)
    ax1.hist(Varray[:,xi_pos], bins=bin_edges,label='Contrôle', edgecolor='black', alpha=0.5, color="black")
    # ax1.legend()

    # Params de l'hist 2
    ax2.set_title(f"Répartition des valeurs\npour la variable {xi},\nconditions initiales perturbées", font="Avenir", size=14)
    ax2.set_xlabel(f"Valeurs de {xi}", font="Avenir", size=14)
    ax2.hist(Parray[:,xi_pos], bins=bin_edges,label='Perturbations', edgecolor='black', alpha=0.5, color="r")
    # ax2.legend()

    plt.tight_layout()
    # plt.savefig(f"../../exp_II_fig/3-1_{xi}_histogrm.png", dpi=300)
    plt.show()

#Création des 3 graphiques :
# for xi in [0,1,2]:
#     xi_hist(values_array, perturb_array, xi)

#---------------------------------------------
#       3.2 - CLIMAT DU SYSTÈME CHAOTIQUE
#          -> TEST STATISTIQUE
#---------------------------------------------
# Utilisation du test à 2 échantillons de Kolmogorov-Smirnov
def ks_func(data1, data2, xi):
    variable = ["x","y","z"][xi]
    ks_2samp_results = scipy.stats.ks_2samp(data1[:,xi], data2[:,xi])
    result=np.array([["variable", variable],
                    ["statistic", ks_2samp_results.statistic],
                    ["pvalue",ks_2samp_results.pvalue],
                    ["statistic location", ks_2samp_results.statistic_location],
                    ["statistic sign", ks_2samp_results.statistic_sign]]
                    )
    return result

for xi in [0,1,2]:
    ks_result=ks_func(values_array,perturb_array,xi)
    print(ks_result)
    ks_array = ks_result if xi==0 else np.concatenate([ks_array,ks_result])
    print()
print(pd.DataFrame(ks_array))

