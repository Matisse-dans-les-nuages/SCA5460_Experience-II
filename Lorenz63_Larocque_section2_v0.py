#=================================
#       IMPORTS
#=================================

import numpy as np
import matplotlib.pyplot as plt
import math

#==================================================================
#       PARTIE 2 - PRÉVISIBILITÉ MOYENNE
#==================================================================
#---------------------------------
#       CONSTANTES
#---------------------------------

DT = 0.001 # Pas de temps delta_t = 0.001 [s]
N = int(4E4) # longueur N de la simulation, en pas de temps
S = 10      # Constante Sigma
B = 2.667   # Constante Beta
R_3 = 28.0  # Constante Rho, solution chaotique
E = 0.1 # Constante epsilon la plus grande

#---------------------------------------------
#       FONCTIONS DES ÉQUATIONS DE BASE
#---------------------------------------------

#Fonction Epsilon
# PARAMS : epsilon (e), facteur d'amplitude (fa)
# RETURNS : Epsilon modifié en fonction de l'amplitude de la perturbation
# On donne à E le facteur fa en exposant, puisque E est une valeur décimale (1/10),
# sa valeur réduira en fonction de l'exposant,
def Epsilon(fa):
    global E
    return E**fa

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
def CI_p(e, r, fa):
    return [e+Epsilon(fa), e+3, r-1]

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
#       POUR SOLUTIONS 2.n
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
#       POUR SOLUTIONS 2.n
#---------------------------------------------
# 1 - Adaptée à x_i(t), où x_i=[x, y, z], avec i=[0, 1, 2]
#
# PARAMS :  values_array (numpy array) : un array donné en fonction de R_3
#           perturb_array (numpy array) : un array donné en fonction de la perturbation epsilon
#           soln (int) :  numéro de la solution à l'étude, soln =[0,1,2]
#           i (int) :  la position de la variable, x, y ou z, l'ordonnée, dans la liste xi
# RETURNS : aucun (graphique créé et enregistré par la fonction)
def xi_t_graphique(values_array,perturb_array, i, Nlen):
    xi=["x","y","z"]
    # Tracé du graphique contrôle, en scatterplot
    plt.scatter(
        range(0, Nlen), # Valeurs de temps [s] dans les abcisses
        values_array[:, i], #Valeurs de x, y ou z dans les ordonnés
        s=0.05,
        c="black",
        label = f"{xi[i]}(t) - contrôle")
    # Tracé du graphique perturbation, en scatterplot
    plt.scatter(
        range(0, Nlen),  # Valeurs de temps [s] dans les abcisses
        perturb_array[:, i],  # Valeurs de x, y ou z dans les ordonnés
        s=0.05,
        c="red",
        label=f"{xi[i]}(t) - perturbation")
    # Label des abcisses
    plt.xlabel("Pas de temps N", font="Avenir", size=14)
    # Label des ordonnés
    plt.ylabel(f"{xi[i]}(t)", font="Avenir", size=14)
    # Titre et sous-titre (superior title, title)
    plt.suptitle(f"2.1 - Prévisibilité moyenne", font="Avenir", size=12)
    plt.title(f"Comparaison de {xi[i]} en fonction du temps\n"
              r"valeurs contrôle et avec perturbation $\epsilon_1$ = 0.01",font="Avenir",size=14)
    plt.legend(markerscale=10)
    #Production et sauvegarde du graphique
    plt.tight_layout()
    # plt.savefig(f"../../exp_II_fig/2-1_{xi[i]}_fn_t.png", dpi=300)
    plt.show()

#---------------------------------------------
#       2.1 - PRÉVISIBILITÉ MOYENNE
#          -> ARRAYS
#---------------------------------------------
values_array = XYZ_array(
        R_3,                    # Argument R_n de XYZ_array()
        CI(Eta(B, R_3), R_3),   # Argument CI de XYZ_array(), avec fonction CI contrôle
        N                       # Argument Nlen de XYZ_array()
    )
perturb_array = XYZ_array(
        R_3,                    # Argument R_n de XYZ_array()
        CI_p(Eta(B, R_3), R_3, 2),   # Argument CI de XYZ_array(), avec fonction CI_p perturbation, donne epsilon = 0.01
        N                       # Argument Nlen de XYZ_array()
    )
#GRAPHIQUES :
# for i in [0, 1, 2]:
#     xi_t_graphique( values_array,perturb_array, i, N) # Autres arguments de xi_t_graphique

#---------------------------------------------
#       2.1 - CALCULS DES VALEURS
#       ÉCARTS_TYPES (std)
#---------------------------------------------
# Écarts-types pour x(t), y(t), z(t), des valeurs de contrôles
x_t_std = values_array[:,0].std()
y_t_std = values_array[:,1].std()
z_t_std = values_array[:,2].std()

# Affichage des écarts-types
print(r"$\sigma_x$",x_t_std,"\n",
    r"$\sigma_y$",y_t_std,"\n",
    r"$\sigma_z$",z_t_std,"\n")

# Calcul de caractérisation de la prévisibilité de la solution
x_combined_subtracted = abs(values_array[:,0]-perturb_array[:,0])
for i,n in enumerate(x_combined_subtracted):
    if n >= x_t_std:
        # comme i correspond à l'indice, i/1000 correspond au pas de temps
        # ( au nombre de secondes écoulées)
        t_prev = i/1000
        print(f"Temps : ({t_prev} x 10^3) N,\ndifférence absolue : {n},\n {1000*t_prev}e pas de temps")
        break

# plt.scatter(10361, 7.653394086214808, s=100, marker="x", c="r")
# plt.plot(range(0,N), x_combined_subtracted)
# plt.show()
#---------------------------------------------
#       2.2 - PRÉVISIBILITÉ SELON
#             L'INCERTITUDE OBSERVATIONNELLE
#        -> CALCULS DES VALEURS
#               std + PRÉVISIBILITÉ
#---------------------------------------------
t_prev_list = [] # Liste des résultats de temps t_prev, indicateur de prévisibilité
perturb_list = [] # Liste des epsilons calculés (magnitude des perturbations)
for fa in range(1,7):
    perturb_array = XYZ_array(
            R_3,                    # Argument R_n de XYZ_array()
            CI_p(Eta(B, R_3), R_3, fa),   # Argument CI de XYZ_array(), avec fonction CI_p perturbation, donne epsilon = 0.1 à 0.000001
            N                       # Argument Nlen de XYZ_array()
        )
    x_combined_subtracted = abs(values_array[:, 0] - perturb_array[:, 0]) #Donne un array de la valeur absolue des différences
    for i, n in enumerate(x_combined_subtracted):
        if n >= x_t_std:
            # comme i correspond à l'indice, i/1000 correspond au pas de temps
            # ( au nombre de secondes écoulées)
            t_prev = i / 1000
            # print(f"Epsilon = {round(Epsilon(fa),fa)}, Temps : {t_prev} [s],\ndifférence absolue : {n},\n {1000 * t_prev}e pas de temps")
            t_prev_list.append(t_prev)
            perturb_list.append(round(Epsilon(fa),fa)) #On arrondit la valeur à la position de 1 après le point décimal.
            break
# Vérification de la complétion des listes de valeurs
# print(t_prev_list, perturb_list)

#---------------------------------------------
#       2.2 - PRÉVISIBILITÉ SELON
#             L'INCERTITUDE OBSERVATIONNELLE
#          -> GRAPHIQUES
#---------------------------------------------
plt.scatter(perturb_list, t_prev_list, s=100, marker="+", color="black")
plt.xscale('log')
plt.suptitle(r"2.2 - Prévisibilité selon l'incertitude observationnelle", font='Avenir', size=12)
plt.title(r"Variations de la prévibilité $\mathit{t_{prev}}$"+"\n"+r"en fonction de la magnitude d'une perturbation $\epsilon$", font="Avenir", size=14)
plt.xlabel(r"Amplitude de la perturbation $\epsilon$", font="Avenir",size=14)
plt.ylabel(r"Prévisibilité $\mathit{t_{prev}}$ [$10^3$ N]",font="Avenir", size=14)

plt.grid(True, which = 'both', linestyle='--', linewidth = 0.5)
plt.tick_params(axis='x', which='major', width=2)

plt.tight_layout()
# plt.savefig(f"../../exp_II_fig/2-2_tprev_fn_e.png", dpi=300)
plt.show()
plt.close() #Graphique complété et sauvegardé

#---------------------------------------------
#       2.3 - PRÉVISIBILITÉ SELON
#             LE RÉGIME DE TEMPS
#          -> GRAPHIQUES
#---------------------------------------------
xyCI_list = [(-2,1),(-4,-1),(-6,-1),(-8,-1)] #Liste des combinaisons de CI en x0 et y0 appliquées à Eta
x0_tPrev_list=[] #Liste des combinaisons de CI en x_0 et de prévisibilité selon tprev

for ci in xyCI_list:
    Varray = XYZ_array(R_3, CI_xyp(e=Eta(B, R_3), r=R_3, xp=ci[0],yp=ci[1], fa=0), N) # Array de valeurs contrôles
    x_0 = float(Varray[0,0]) #Valeur initiale en x
    Parray = XYZ_array(R_3, CI_xyp(e=Eta(B, R_3), r=R_3, xp=ci[0],yp=ci[1], fa=2), N) # Array avec perturbation
    x_t_std = Varray[:, 0].std() # Valeur de l'écart-type

    # Calcul de caractérisation de la prévisibilité de la solution
    x_combined_subtracted = abs(Varray[:, 0] - Parray[:, 0]) #Donne un array de la valeur absolue des différences
    for i, n in enumerate(x_combined_subtracted):
        if n >= x_t_std:
            # comme i correspond à l'indice, i/1000 correspond au pas de temps
            # ( au nombre de secondes écoulées)
            t_prev = i / 1000
            # print(f"Temps : {t_prev} [s],\ndifférence absolue : {n},\n {1000*t_prev}e pas de temps")
            x0_tPrev_list.append([x_0, t_prev])
            break
x0_tPrev_array=np.array(x0_tPrev_list)
print(x0_tPrev_array)
#Mêmes configurations graphiques que précédemment
plt.scatter(x0_tPrev_array[:,0], x0_tPrev_array[:,1], marker="+", s=100, color="black")
#Titre
plt.suptitle("2.3 - Prévibilité moyenne selon le régime de temps", font="Avenir", size=12)
plt.title(r"Variabilité de la prévisibilité $\mathit{t_{prev}}$"+
          "\n"+r" selon la condition initiale $x_0$", font="Avenir",size=14)
#Label et gestion des axes
plt.xlabel(r"Condition initiale $x_0$", font="Avenir",size=14)
plt.ylabel(r"Prévisibilité $\mathit{t_{prev}}$ [$10^3$ N]", font="Avenir",size=14)
#ajout de ticks pour lisibilité des valeurs
plt.grid(True, which = 'both', linestyle='--', linewidth = 0.5)
plt.tick_params(axis='x', which='major', width=2)
#Gestion de l'Espace
plt.tight_layout()
# plt.savefig(f"../../exp_II_fig/2-3_tprev_fn_x0.png", dpi=300)
plt.show()
