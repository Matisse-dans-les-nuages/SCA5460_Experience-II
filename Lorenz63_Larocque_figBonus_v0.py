#=================================
#       IMPORTS
#=================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy

#==================================================================
#       PARTIE 4 - Figure Bonus
#==================================================================
#---------------------------------
#       CONSTANTES
#---------------------------------

DT = 0.001 # Pas de temps delta_t = 0.001 [s]
# N = int(1E5) # longueur N de la simulation, en pas de temps
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
#       POUR SOLUTIONS - Bonus
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
    for n in range(0,int(Nlen-1)):
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

# Utilisation du test à 2 échantillons de Kolmogorov-Smirnov
def ks_func(data1, data2, xi):
    # variable = ["x","y","z"][xi]
    ks_2samp_results = scipy.stats.ks_2samp(data1[:,xi], data2[:,xi])
    return ks_2samp_results.pvalue

#---------------------------------------------
#       Bonus - CLIMAT DU SYSTÈME CHAOTIQUE
#      -> Mesure stat de la prévisibilité
#---------------------------------------------
#Créer la liste des N échelles de temps
N_list=[]

for i in range(2,6):
    if i > 4:
        # Pour les échelles de temps supérieures à 10,
        # on vérifie les quartiles des ordres de grandeur aussi
        N_list.append((10 ** i) / 8) # 1/8
        N_list.append((10 ** i) / 4) # 1/4
        N_list.append(((10 ** i) / 4) + ((10 ** i) / 8)) # 3/8
        N_list.append((10 ** i) / 2) #1/2
        N_list.append(((10 ** i) / 2) + ((10 ** i) / 8)) # 5/8
        N_list.append(((10 ** i) / 2) + ((10 ** i) / 4)) #3/4
        N_list.append(((10 ** i) / 2) + ((10 ** i) / 4)+((10 ** i) / 8)) #7/8
        N_list.append(10 ** i) # 1
    elif i > 1:
        # Pour les échelles de temps supérieures à 10,
        # on vérifie les quartiles des ordres de grandeur aussi
        N_list.append((10 ** i) / 4)
        N_list.append((10 ** i) / 2)
        N_list.append(((10 ** i) / 2)+((10 ** i) / 4))
        N_list.append(10 ** i)


print( N_list)
#Data des pvalues pour le test ks
x_KSpVal_list=[]
y_KSpVal_list=[]
z_KSpVal_list=[]

x_data=[]
y_data=[]
z_data=[]

x_pdata=[]
y_pdata=[]
z_pdata=[]

for Nstep in N_list:
    values_array = XYZ_array(
        R,  # Argument R_n de XYZ_array()
        CI(Eta(B, R), R),  # Argument CI de XYZ_array(), avec fonction CI contrôle
        Nstep ) # Argument Nlen de XYZ_array()
    perturb_array = XYZ_array(
        R,  # Argument R_n de XYZ_array()
        CI_p(e=Eta(B, R), r=R),  # Argument CI de XYZ_array(), avec fonction CI_p perturbation, epsilon = 0.01
        Nstep ) # Argument Nlen de XYZ_array()
    print(Nstep)
    for xi in [0, 1, 2]:
        ks_result = ks_func(values_array, perturb_array, xi) #On obtient ici le pvalue, dans un axe x, y ou z, en fonction de l'échelle de temps N
        [x_KSpVal_list,y_KSpVal_list,z_KSpVal_list][xi].append(round(float(ks_result),6))
        if Nstep==(1E5/4):
            [x_data,y_data,z_data][xi].append(values_array[:,xi])
            [x_pdata,y_pdata,z_pdata][xi].append(perturb_array[:,xi])




print(x_data)
print(y_data)
print(z_data)

#Sauvegarde des résultats vers un dataframe, pour travailler la figure plus facilement
#Données du test Kolmogorov-Smirnov
pVal_df = pd.DataFrame({
    'N':N_list,
    'x_KSpVal':x_KSpVal_list,
    'y_KSpVal':y_KSpVal_list,
    'z_KSpVal':z_KSpVal_list,
})
pVal_df.to_csv('figBonus_test_df.csv', index=False)

#Données des contrôles à 2.5E5 pas de temps, pour figure 3D
var_df=pd.DataFrame({
    'x': x_data[0],
    'y': y_data[0],
    'z': z_data[0]
})
var_df.to_csv('figBonus_3d_df.csv', index=False)

#Données des perturbations à 2.5E5 pas de temps, pour figure 3D
pvar_df=pd.DataFrame({
    'x': x_pdata[0],
    'y': y_pdata[0],
    'z': z_pdata[0]
})
pvar_df.to_csv('figBonus_3d_p_df.csv', index=False)