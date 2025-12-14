import os
import math
import random
import datetime
import glob
from pathlib import Path
from collections import deque

# Créer le dossier data (au même niveau que src/) s'il n'existe pas
base_dir = Path(__file__).resolve().parent.parent  # .../src
data_dir = base_dir / "data"
data_dir.mkdir(exist_ok=True)

def generer_csv(nom_fichier, nb_points=1000):
    """
    Génère un fichier CSV avec une forme d'onde complexe combinant plusieurs fréquences,
    du bruit, des pics aléatoires et une dérive par paliers.
    Format: date;heure;valeur;flag
    """
    # Créer le chemin complet du fichier
    chemin_fichier = data_dir / nom_fichier
    
    # Date de départ
    date_courante = datetime.datetime(2025, 3, 1, 0, 0, 0)
    
    # Paramètres des ondes - 7 composantes harmoniques
    amplitude_totale = 6

    # Sept périodes : 60, 70, 84, 105, 140, 210, 420
    periode1 = 420
    periode2 = 210
    periode3 = 140
    periode4 = 105
    periode5 = 84
    periode6 = 70
    periode7 = 60

    # Calcul de la période complète du signal (PPCM simplifié)
    periode_complete = 60

    # Fréquences correspondantes
    frequence1 = 1 / periode1
    frequence2 = 1 / periode2
    frequence3 = 1 / periode3
    frequence4 = 1 / periode4
    frequence5 = 1 / periode5
    frequence6 = 1 / periode6
    frequence7 = 1 / periode7

    # Répartition des amplitudes (équilibrée)
    amplitude1 = amplitude_totale * 1.00  # 420 → 100%
    amplitude2 = amplitude_totale * 0.85  # 210 → 85%
    amplitude3 = amplitude_totale * 0.70  # 140 → 70%
    amplitude4 = amplitude_totale * 0.60  # 105 → 60%
    amplitude5 = amplitude_totale * 0.50  # 84 → 50%
    amplitude6 = amplitude_totale * 0.40  # 70 → 40%
    amplitude7 = amplitude_totale  # 60 → 30%
    
    # Paramètres pour les bruits et pics
    bruit_amplitude = 0.05
    pic_amplitude = 1
    proba_pic = 0.01        # 0.5% de probabilité d'avoir un pic

    # Configuration des zones de dérive (6 zones réparties)
    # Les 4 premières périodes sont propres (sans dérive ni anomalie)
    # Format: (debut_periode, fin_periode, type_derive, amplitude_derive)
    zones_derive = [
        (6, 11, "exponentielle", 5.0),     # Dérive 1: exponentielle montante
        (14, 19, "lineaire", -4.0),        # Dérive 2: linéaire descendante
        (24, 30, "paliers", 6.0),          # Dérive 3: paliers abrupts
        (37, 44, "sinusoidale", 4.5),      # Dérive 4: sinusoïdale lente
        (52, 60, "polynomiale", 5.5),      # Dérive 5: polynomiale (quadratique)
        (64, 71, "combinee", 6.5),         # Dérive 6: combinée (marche + oscillations)
    ]

    # Seuil de variation pour marquer comme erreur: 20% de l'amplitude totale
    seuil_erreur_variation = 0.20 * amplitude_totale

    # Pourcentage de la période complète à marquer en erreur lors de grosses dérives
    pct_erreur_forcee_derive = 0.05  # 5% de la période complète
    
    # Historique des valeurs récentes pour le calcul de tendance
    historique_valeurs = deque(maxlen=10)
    
    # Liste pour stocker toutes les données avant de les écrire
    toutes_donnees = []
    
    for i in range(nb_points):
        # Calcul de la valeur combinant 7 sinusoïdes de périodes différentes
        t = i
        valeur1 = amplitude1 * math.sin(2*math.pi*frequence1 * t + 0.0)
        valeur2 = amplitude2 * math.sin(2*math.pi*frequence2 * t + 0.5)
        valeur3 = amplitude3 * math.sin(2*math.pi*frequence3 * t + 1.2)
        valeur4 = amplitude4 * math.sin(2*math.pi*frequence4 * t + 2.3)
        valeur5 = amplitude5 * math.sin(2*math.pi*frequence5 * t + 3.1)
        valeur6 = amplitude6 * math.sin(2*math.pi*frequence6 * t + 0.8)
        valeur7 = amplitude7 * math.sin(2*math.pi*frequence7 * t + 1.7)

        # Combinaison des sinusoïdes (toutes les 7 composantes)
        valeur_base =  valeur7
        valeur_base = (valeur_base + amplitude_totale * 2.5) / 2

        # Ajout de bruit
        bruit = random.uniform(-bruit_amplitude, bruit_amplitude) * amplitude_totale

        # Calcul de la dérive en fonction des zones définies
        derive = 0
        periode_actuelle = i / periode_complete
        dans_zone_derive = False
        amplitude_derive_actuelle = 0
        debut_derive_early = False

        for debut_p, fin_p, type_derive, amplitude_derive in zones_derive:
            if debut_p <= periode_actuelle < fin_p:
                dans_zone_derive = True
                amplitude_derive_actuelle = amplitude_derive

                # Position relative dans la zone de dérive (0 à 1)
                pos_zone = (periode_actuelle - debut_p) / (fin_p - debut_p)
                early_window_in_periods = 0  # 50% d'une période
                if periode_actuelle < debut_p + early_window_in_periods:
                    debut_derive_early = True

                if type_derive == "exponentielle":
                    # Dérive exponentielle montante
                    derive = amplitude_derive * (math.exp(2.5 * pos_zone) - 1) / (math.exp(2.5) - 1)

                elif type_derive == "lineaire":
                    # Dérive linéaire (peut être négative)
                    derive = amplitude_derive * pos_zone

                elif type_derive == "paliers":
                    # Paliers abrupts (5 paliers)
                    nb_paliers = 5
                    palier = int(pos_zone * nb_paliers) / nb_paliers
                    derive = amplitude_derive * palier

                elif type_derive == "sinusoidale":
                    # Dérive sinusoïdale lente (1-2 oscillations complètes)
                    derive = amplitude_derive * math.sin(2 * math.pi * pos_zone * 1.5)

                elif type_derive == "polynomiale":
                    # Dérive polynomiale (quadratique)
                    derive = amplitude_derive * (pos_zone ** 2)

                elif type_derive == "combinee":
                    # Dérive combinée: marche d'escalier + oscillations
                    palier = int(pos_zone * 3) / 3
                    oscillation = 0.3 * math.sin(2 * math.pi * pos_zone * 4)
                    derive = amplitude_derive * (palier + oscillation)

                break  # On ne prend en compte que la première zone trouvée

        # Ajout potentiel d'un pic (seulement après 4 périodes propres)
        pic = 0
        a_un_pic = False
        if periode_actuelle >= 4.0 and random.random() < proba_pic:
            pic = random.uniform(-pic_amplitude, pic_amplitude) * amplitude_totale
            a_un_pic = True

        # Valeur finale
        valeur = valeur_base + bruit + pic + derive

        # Détection des anomalies basée sur la variation
        flag = 0
        if dans_zone_derive and debut_derive_early:
            flag = 1
        # 1. Détection des pics (variation > 20% de l'amplitude totale)
        if a_un_pic and abs(pic) > seuil_erreur_variation:
            flag = 1

        # 2. Erreurs forcées lors de grosses dérives (5% de la période dans chaque zone)
        # if dans_zone_derive and abs(amplitude_derive_actuelle) >= 4.0:
        #     # Marquer aléatoirement 5% des points dans cette zone comme erreurs
        #     if random.random() < pct_erreur_forcee_derive:
        #         flag = 1

        # Ajouter à l'historique pour calculer la tendance
        historique_valeurs.append(valeur)
        
        # Formatage de la date et de l'heure
        date_str = date_courante.strftime("%d/%m/%Y")
        heure_str = date_courante.strftime("%H:%M:%S")
        
        # Stocker les données
        toutes_donnees.append({
            "date": date_str,
            "heure": heure_str,
            "valeur": valeur,
            "flag": flag,
            "a_supprimer": False  # Sera utilisé plus tard
        })
        
        # Incrémentation de la date
        date_courante += datetime.timedelta(seconds=30)

    # 1. Suppression de groupes de points consécutifs et marquage du point suivant comme erreur
    nb_sequences_a_supprimer = 10  # Créer 10 "trous" temporels
    longueur_min_sequence = 3  # Minimum 3 points consécutifs à supprimer
    longueur_max_sequence = 15  # Maximum 15 points consécutifs à supprimer

    # Points de départ possibles (éviter les 4 premières périodes propres)
    debut_suppression = int(4 * periode_complete)
    plage_valide = range(debut_suppression, nb_points - longueur_max_sequence - 10)

    # Sélectionner les points de départ des séquences à supprimer
    points_depart = sorted(random.sample(plage_valide, nb_sequences_a_supprimer))

    # Supprimer les séquences et marquer les points suivants comme erreurs
#     for point_depart in points_depart:
#         # Déterminer la longueur de la séquence à supprimer
#         longueur_sequence = random.randint(longueur_min_sequence, longueur_max_sequence)
#         
#         # Marquer tous les points de la séquence pour suppression
#         for i in range(longueur_sequence):
#             if point_depart + i < nb_points:
#                 toutes_donnees[point_depart + i]["a_supprimer"] = True
#         
#         # Marquer le point suivant la séquence comme erreur (s'il existe et n'est pas déjà à supprimer)
#         point_suivant = point_depart + longueur_sequence
#         if point_suivant < nb_points and not toutes_donnees[point_suivant]["a_supprimer"]:
#             toutes_donnees[point_suivant]["flag"] = 1  # Marquer comme erreur
#             print(f"Groupe de {longueur_sequence} points supprimés à partir de l'index {point_depart}, point {point_suivant} marqué comme anomalie")
# # Écrire les données dans le fichier en excluant les points à supprimer
    with open(chemin_fichier, 'w', encoding='utf-8') as f:
        for donnee in toutes_donnees:
            if not donnee["a_supprimer"]:
                ligne = f"{donnee['date']};{donnee['heure']};{donnee['valeur']:.8f};{donnee['flag']}\n"
                f.write(ligne)

# Supprimer les fichiers test_* existants
fichiers_a_supprimer = glob.glob(str(data_dir / "test_*"))
for fichier in fichiers_a_supprimer:
    os.remove(fichier)
    print(f"Fichier supprimé : {fichier}")
print(f"Nombre total de fichiers supprimés : {len(fichiers_a_supprimer)}")

# Générer plusieurs fichiers CSV
for i in range(1, 6):
    nom_fichier = f"test_{i}.csv"
    
    generer_csv(nom_fichier)
    print(f"Fichier {nom_fichier} généré dans le dossier data/")
