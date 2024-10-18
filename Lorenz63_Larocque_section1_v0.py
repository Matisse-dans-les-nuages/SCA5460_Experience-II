#=================================
#       IMPORTS
#=================================

import numpy as np
import matplotlib.pyplot as plt
import math

#==================================================================
#       PARTIE 1 - SOLUTION NUMÉRIQUE DU SYSTÈME
#==================================================================
#---------------------------------
#       CONSTANTES
#---------------------------------

DT = 0.001 # Pas de temps delta_t = 0.001 [s]
N = int(2E4) # longueur N de la simulation, en pas de temps
S = 10      # Constante Sigma
B = 2.667   # Constante Beta
R_1 = 1.0   # Constante Rho, solution stationnaire
R_2 = 350.0 # Constante Rho, solution périodique
R_3 = 28.0  # Constante Rho, solution chaotique

#---------------------------------------------
#       FONCTIONS DES ÉQUATIONS DE BASE
#---------------------------------------------

# Fonction Eta
# PARAMS : beta (b), rho (r), applicables selon la solution (1.n)
# RETURN : eta, en fonction de l'équation du devoir, partie 1
def Eta(b,r):
    return math.sqrt(b*(r-1))

# Fonction CI
# PARAMS : eta (e), rho (r), applicables selon la solution (1.n)
# RETURN : une liste avec x_t0, y_t0, z_t0, les conditions initiales au temps 0 pour x, y, z.
def CI(e, r):
    return [e, e+3, r-1]

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
#       POUR SOLUTIONS 1.n
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
#       FONCTIONS GRAPHIQUES
#       POUR SOLUTIONS 1.n
#---------------------------------------------
# 1 - Adaptée à x_i(t), où x_i=[x, y, z], avec i=[0, 1, 2]
#
# PARAMS :  values_array (numpy array) : un array donné en fonction de R_1, R_2, ou R_3
#           soln (int) :  numéro de la solution à l'étude, soln =[0,1,2]
#           i (int) :  la position de la variable, x, y ou z, l'ordonnée, dans la liste xi
# RETURNS : aucun (graphique créé et enregistré par la fonction)
def xi_t_graphique(values_array, soln, i, Nlen):
    xi=["x","y","z"]
    sol_list=["stationnaire", "périodique","chaotique"]
    # Tracé du graphique, en scatterplot
    plt.scatter(
        range(0, Nlen), # Valeurs de temps [s] dans les abcisses
        values_array[:, i], #Valeurs de x, y ou z dans les ordonnés
        s=0.05,
        c="black")
    # Label des abcisses
    plt.xlabel("Pas de temps N", font="Avenir", size=14)
    # Label des ordonnés
    plt.ylabel(f"{xi[i]}(N)", font="Avenir", size=14)
    # Titre et sous-titre (superior title, title)
    plt.suptitle(f"1.{1+soln} - Solution {sol_list[soln]}", font="Avenir", size=12)
    plt.title(f"Variations de {xi[i]} en fonction du temps",font="Avenir",size=14)
    #Production et sauvegarde du graphique
    plt.tight_layout()
    # plt.savefig(f"../../exp_II_fig/1-{soln}_{xi[i]}_fn_t.png", dpi=300)
    plt.show()

# 2 - Adaptée à x(x_i), où x_i=[y, z], avec i=[0, 1]
#
# PARAMS : values_array (numpy array) - un array donné en fonction de R_1, R_2, ou R_3
#          soln (int) -  numéro de la solution à l'étude, soln =[0,1,2]
#          i (int) -  la position de la variable, y ou z, l'ordonnée, dans la liste xi
# RETURNS : aucun (graphique créé et enregistré par la fonction)
def x_xi_graphique(values_array, soln, i):
    xi=["y","z"]
    sol_list=["stationnaire", "périodique","chaotique"]
    # Tracé du graphique, en scatterplot
    plt.scatter(
        values_array[:, i+1], # Valeurs de y ou de z dans les abcisses
        values_array[:, 0], # Valeurs de x dans les ordonnés
        s=0.05,
        c="black")
    # Label des abcisses
    plt.xlabel(f"Valeurs de {xi[i]}", font="Avenir", size=14)
    # Label des ordonnés
    plt.ylabel(f"x({xi[i]})", font="Avenir", size=14)
    #Titre et sous-titre (superior title, title)
    plt.suptitle(f"1.{1+soln} - Solution {sol_list[soln]}", font="Avenir", size=12)
    plt.title(f"Variations de x en fonction de {xi[i]}",font="Avenir",size=14)
    #Production et sauvegarde du graphique
    plt.tight_layout()
    # plt.savefig(f"../../exp_II_fig/1-{soln}_x_fn_{xi[i]}.png", dpi=300)
    plt.show()
#---------------------------------------------
#       1.N - SOLUTIONS 1, 2, 3
#          -> GRAPHIQUES
#---------------------------------------------
rho_tuple=[R_1, R_2, R_3]
for soln, R_n in enumerate(rho_tuple):
    #Ici, les index de la fonction enumerate() jouent le rôle des numéros 1, 2 et 3 des solutions (l'argument soln)
    # Graphiques de x, y, z en fonciton du temps
    for i in [0, 1, 2]:
        xi_t_graphique( XYZ_array( #début de l'argument values_array de xi_t_graphique()
            R_n,                    # Argument R_n de XYZ_array()
            CI(Eta(B, R_n), R_n),   # Argument CI de XYZ_array()
            N                       # Argument Nlen de XYZ_array()
        ), #Fin de l'argument values_array de xi_t_graphique()
            soln, i, N) # Autres arguments de xi_t_graphique

    # Graphiques de x ne fonction de y et z
    for i in [0, 1]:
        #Même ordre des arguments
        x_xi_graphique( XYZ_array(
            R_n,
            CI(Eta(B, R_n), R_n),
            N
        ), #Fin de l'argument values_array de x_xi_graphique()
            soln, i) # Autres arguments de x_xi_graphique
