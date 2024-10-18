#=================================
#       IMPORTS
#=================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

#==================================================================
#       PARTIE 4 - Figure Bonus
#==================================================================

#---------------------------------
#       DATAFRAME, GRAPHIQUE
#       KS
#---------------------------------
pVal_df = pd.read_csv('figBonus_test_df.csv')

#DATA - Valeurs N
n_val = np.array(pVal_df['N'][:])

#DATA - Kolomogorov - Smirnov
x_KSval = np.array(pVal_df['x_KSpVal'][:])
y_KSval = np.array(pVal_df['y_KSpVal'][:])
z_KSval = np.array(pVal_df['z_KSpVal'][:])
dataKS=[x_KSval, y_KSval, z_KSval]

#GRAPHIQUE - données répétitives automatisées
var = ["x","y","z"] #Nom des variables, pour légendes
# markers = ["x","+","D"] #Marqueurs, changeants à chaque variable xi = (x,y,z)
colors = ["#444444", "#e7004e", "#29cdf3" ] #Couleurs, changeantes à chaque variable xi = (x,y,z)

#GRAPHIQUE
# Titre et sous-titre (superior title, title)
plt.suptitle(f"figure bonus", font="Avenir", size=12)

# ax1 - p-values du test Kolmogorov-Smirnov
for xi in [0,1,2]:
    plt.plot(n_val,
             dataKS[xi],
             label = f"p-value, {var[xi]}",
             c=colors[xi])
    plt.scatter(n_val,
            dataKS[xi],
            c=colors[xi],
            s=50,
            marker="+")


#Titre
plt.title(f"P-value en fonction du pas de temps,"
          f"\nà partir des simulations contrôles et perturbées"
          f"\n(test Kolmogorov-Smirnov)",font="Avenir",size=14)

#Ligne seuil p = 0.05
plt.plot(n_val, np.full(len(n_val), 0.05), color="#ffc208", label="p-value = 0.05")
plt.xscale('log')

#Titres des axes, ax1
plt.xlabel("Pas de temps N", font="Avenir", size=14)
plt.ylabel(f"p-value (N)", font="Avenir", size=14)

#Légende, ax1
plt.legend(markerscale=1)
plt.grid(True, which = 'both', linestyle='--', linewidth = 0.5)
plt.tick_params(axis='x', which='major', width=2)

#Production et sauvegarde du graphique
plt.tight_layout()
# plt.savefig(f"../../exp_II_fig/bonusFig_pval_fn_N.png", dpi=300)
plt.show()

#---------------------------------
#       DATAFRAME, GRAPHIQUE
#       3D - contrôle
#---------------------------------
xi_var_df = pd.read_csv("figBonus_3d_df.csv")

x_var = np.array(xi_var_df['x'][:])
y_var = np.array(xi_var_df['y'][:])
z_var = np.array(xi_var_df['z'][:])

# Figure 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#DATA : Les données x, y, z à N=(1E5/4)
ax.scatter(x_var, y_var, z_var, c='#444444', marker='o', s=0.25)

#Titres des axes
ax.set_xlabel('x(x,y)')
ax.set_ylabel('y(x,y,z)')
ax.set_zlabel('z(y,z))')
ax.set_title('Représentation 3D de la solution chaotique pour N = 2.5E5')

#Projections 2D:
# XY
ax.scatter(x_var, y_var, np.zeros_like(z_var), c='#e7004e', marker='o', alpha=0.05, label='XY projection',s=0.25)

# XZ
ax.scatter(x_var, np.zeros_like(y_var)+20, z_var, c='#29cdf3', marker='o', alpha=0.05, label='XZ projection',s=0.25)

# YZ
ax.scatter(np.zeros_like(x_var)-20, y_var, z_var, c='#ffc208', marker='o', alpha=0.05, label='YZ projection',s=0.25)

# Affichage
plt.show()

#---------------------------------
#       DATAFRAME, GRAPHIQUE
#       3D - perturbations
#---------------------------------
xi_var_df = pd.read_csv("figBonus_3d_p_df.csv")

x_var = np.array(xi_var_df['x'][:])
y_var = np.array(xi_var_df['y'][:])
z_var = np.array(xi_var_df['z'][:])

# Figure 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#DATA : Les données x, y, z à N=(1E5/4)
ax.scatter(x_var, y_var, z_var, c='#444444', marker='o', s=0.25)

#Titres des axes
ax.set_xlabel('x(x,y)')
ax.set_ylabel('y(x,y,z)')
ax.set_zlabel('z(y,z))')
ax.set_title('Représentation 3D de la solution chaotique pour N = 2.5E5')

#Projections 2D:
# XY
ax.scatter(x_var, y_var, np.zeros_like(z_var), c='#e7004e', marker='o', alpha=0.05, label='XY projection',s=0.25)

# XZ
ax.scatter(x_var, np.zeros_like(y_var)+20, z_var, c='#29cdf3', marker='o', alpha=0.05, label='XZ projection',s=0.25)

# YZ
ax.scatter(np.zeros_like(x_var)-20, y_var, z_var, c='#ffc208', marker='o', alpha=0.05, label='YZ projection',s=0.25)

# Affichage
plt.show()

