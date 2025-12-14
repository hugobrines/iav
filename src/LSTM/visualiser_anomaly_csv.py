import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import argparse
from matplotlib.widgets import Slider, Button

class AnomalyResultVisualizer:
    def __init__(self, original_csv_path, analysis_csv_path, start_point=None):
        """
        Initialise le visualiseur de résultats d'anomalies
        
        Args:
            original_csv_path: Chemin vers le fichier CSV original au format
                               Date;Heure;Valeur;Flag (1=vraie anomalie, 0=normal)
            analysis_csv_path: Chemin vers le fichier CSV d'analyse au format
                               Date;Heure;ValeurOriginale;Prediction;AnomalyScore;SeuilAnomalie;EstAnomalie
        """
        self.df_original = None
        self.df_analysis = None
        self.df_merged = None
        self.df_combined = None
        self.start_point = start_point 
        self.current_threshold = None  # Pour stocker le seuil actuel
        self.optimal_threshold = None  # Pour stocker le seuil optimal
        self.load_data(original_csv_path, analysis_csv_path)
        
        # Pour stocker les figures et axes pour l'interactivité
        self.fig_results = None
        self.ax_values = None
        self.ax_scores = None
        self.fig_confusion = None
        self.ax_confusion = None
        self.ax_metrics = None
        
        # Pour stocker les éléments du graphique qui seront mis à jour
        self.scatter_tp_values = None
        self.scatter_fp_values = None
        self.scatter_fn_values = None
        self.scatter_tp_scores = None
        self.scatter_fp_scores = None
        self.scatter_fn_scores = None
        self.threshold_line = None
        self.optimal_threshold_line = None
        self.optimal_threshold_annotation = None
        self.confusion_heatmap = None
        self.metrics_table = None
        self.threshold_slider = None  # Pour stocker le slider

    def load_data(self, original_csv_path, analysis_csv_path):
        """Charge les données depuis les fichiers CSV et les fusionne"""
        # Chargement du fichier original
        self.df_original = pd.read_csv(original_csv_path, sep=';', header=None)
        if len(self.df_original.columns) == 4:
            self.df_original.columns = ['Date', 'Heure', 'Valeur', 'VraieAnomalie']
        else:
            raise ValueError("Le fichier original doit avoir 4 colonnes: Date;Heure;Valeur;Flag")
        
        # Conversion des dates
        self.df_original['Datetime'] = pd.to_datetime(self.df_original['Date'] + ' ' + self.df_original['Heure'], 
                                                    format='%d/%m/%Y %H:%M:%S')
        
        # Chargement du fichier d'analyse
        self.df_analysis = pd.read_csv(analysis_csv_path, sep=';')
        
        # Conversion des dates
        self.df_analysis['Datetime'] = pd.to_datetime(self.df_analysis['Date'] + ' ' + self.df_analysis['Heure'], 
                                                    format='%d/%m/%Y %H:%M:%S')
        
        # Conversion des flags en entiers
        self.df_original['VraieAnomalie'] = self.df_original['VraieAnomalie'].astype(int)
        self.df_analysis['EstAnomalie'] = self.df_analysis['EstAnomalie'].astype(int)
        
        # Appliquer le filtrage selon le point de départ si spécifié
        if self.start_point is not None:
            if isinstance(self.start_point, int):
                # Si c'est un index numérique
                if self.start_point < len(self.df_original):
                    self.df_original = self.df_original.iloc[self.start_point:]
                    self.df_analysis = self.df_analysis.iloc[self.start_point:]
                    print(f"Données filtrées à partir de l'index {self.start_point}")
                else:
                    print(f"ATTENTION: L'index {self.start_point} dépasse la taille des données ({len(self.df_original)} points)")
            elif isinstance(self.start_point, str):
                # Si c'est une date sous forme de chaîne
                try:
                    start_datetime = pd.to_datetime(self.start_point, format='%d/%m/%Y %H:%M:%S')
                    self.df_original = self.df_original[self.df_original['Datetime'] >= start_datetime]
                    self.df_analysis = self.df_analysis[self.df_analysis['Datetime'] >= start_datetime]
                    print(f"Données filtrées à partir de la date {self.start_point}")
                except:
                    print(f"ATTENTION: Format de date incorrect. Utilisation de toutes les données.")
        
        # Fusion des deux dataframes sur la base des timestamps
        self.df_merged = pd.merge(self.df_original, self.df_analysis, on=['Datetime'], suffixes=('_orig', '_analysis'))
        
        
        # Vérifier si les valeurs correspondent
        value_diff = abs(self.df_merged['Valeur'] - self.df_merged['ValeurOriginale']).mean()
        if value_diff > 0.001:
            print(f"ATTENTION: Différence moyenne entre les valeurs des deux fichiers: {value_diff}")
            print("Les fichiers pourraient ne pas correspondre au même jeu de données.")
        
        # Stocker le seuil d'anomalie initial
        self.current_threshold = self.df_analysis['SeuilAnomalie'].iloc[0]
        
        # Calculer les résultats de détection initiaux
        self.update_detection_results(self.current_threshold)
        
        print(f"Données chargées et fusionnées: {len(self.df_merged)} lignes")
        print(f"Seuil d'anomalie initial: {self.current_threshold}")
    
    def update_detection_results(self, threshold):
        """Met à jour les résultats de détection en fonction du seuil fourni"""
        # Mettre à jour le seuil actuel
        self.current_threshold = threshold
        
        # Recalculer les drapeaux d'anomalie détectée
        self.df_merged['DetectedAnomaly'] = self.df_merged['AnomalyScore'] > threshold
        self.df_merged['DetectedAnomaly'] = self.df_merged['DetectedAnomaly'].astype(int)
        
        if hasattr(self, 'df_combined') and self.df_combined is not None:
            # Mettre à jour également dans df_combined si disponible
            real_indices = self.df_combined['IsArtificial'] == 0
            self.df_combined.loc[real_indices, 'DetectedAnomaly'] = self.df_merged['DetectedAnomaly'].values

    def detect_and_add_missing_points(self):
        """
        Détecte les écarts temporels importants et ajoute des points artificiels
        pour visualiser les données manquantes
        """
        if self.df_merged is None or len(self.df_merged) < 5:
            print("Pas assez de données pour détecter les points manquants")
            return
        
        # Calculer le pas de temps normal à partir des 5 premiers points
        time_diffs = []
        for i in range(1, 5):
            diff = (self.df_merged['Datetime'].iloc[i] - self.df_merged['Datetime'].iloc[i-1]).total_seconds()
            time_diffs.append(diff)
        
        # Utiliser la médiane pour plus de robustesse
        normal_time_step = np.median(time_diffs)
        print(f"Pas de temps normal détecté: {normal_time_step} secondes")
        
        # Liste pour stocker les points artificiels
        artificial_points = []
        
        # Parcourir tous les points à partir du second
        for i in range(1, len(self.df_merged)):
            current_time = self.df_merged['Datetime'].iloc[i]
            previous_time = self.df_merged['Datetime'].iloc[i-1]
            time_gap = (current_time - previous_time).total_seconds()
            
            # Vérifier si l'écart est plus de 2 fois le pas normal
            if time_gap > 2 * normal_time_step:
                # Nombre de points manquants à insérer
                num_missing = int(time_gap / normal_time_step) - 1
                print(f"Écart détecté: {time_gap}s - Ajout de {num_missing} points artificiels")
                
                # Valeur du point précédent à répliquer
                prev_value = self.df_merged['Valeur'].iloc[i-1]
                prev_prediction = self.df_merged['Prediction'].iloc[i-1]
                
                # Créer et ajouter les points artificiels
                for j in range(1, num_missing + 1):
                    # Calculer le timestamp du point artificiel
                    artificial_time = previous_time + pd.Timedelta(seconds=j * normal_time_step)
                    
                    # Créer un dictionnaire pour ce point artificiel
                    point = {
                        'Datetime': artificial_time,
                        'Valeur': prev_value,
                        'Prediction': prev_prediction,
                        'AnomalyScore': 0,  # Pas d'anomalie pour les points artificiels
                        'EstAnomalie': 0,
                        'VraieAnomalie': 0,
                        'DetectedAnomaly': 0,
                        'IsArtificial': 1  # Marquer comme artificiel
                    }
                    artificial_points.append(point)
        
        # Ajouter une colonne 'IsArtificial' à df_merged (0 = point réel, 1 = point artificiel)
        self.df_merged['IsArtificial'] = 0
        
        # Convertir la liste de points artificiels en DataFrame
        if artificial_points:
            df_artificial = pd.DataFrame(artificial_points)
            
            # Concaténer les points artificiels avec les points originaux
            self.df_combined = pd.concat([self.df_merged, df_artificial], ignore_index=True)
            
            # Trier par ordre chronologique
            self.df_combined = self.df_combined.sort_values('Datetime').reset_index(drop=True)
            
            print(f"Ajout de {len(artificial_points)} points artificiels pour visualisation")
        else:
            self.df_combined = self.df_merged.copy()
            print("Aucun écart temporel important détecté")

    def update_plots(self, threshold):
        """Met à jour les graphiques en fonction du nouveau seuil"""
        # Mettre à jour les résultats de détection avec le nouveau seuil
        self.update_detection_results(threshold)
        
        # Mettre à jour les graphiques de résultats
        self.update_results_plot()
        
        # Mettre à jour la matrice de confusion et les métriques
        self.update_confusion_matrix()
        
        # Redessiner les figures
        if self.fig_results:
            self.fig_results.canvas.draw_idle()
        if self.fig_confusion:
            self.fig_confusion.canvas.draw_idle()
    
    def update_results_plot(self):
        """Met à jour le graphique des résultats sans le recréer entièrement"""
        if not hasattr(self, 'df_combined') or self.df_combined is None:
            return
            
        # Définir les couleurs pour les différents types de points
        colors = {
            'true_positive': 'red',           # Détecté comme anomalie + Est réellement une anomalie
            'false_positive': 'orange',       # Détecté comme anomalie + N'est pas une anomalie
            'true_negative': 'green',         # Non détecté comme anomalie + N'est pas une anomalie
            'false_negative': 'magenta',      # Non détecté comme anomalie + Est une anomalie
            'artificial': 'purple'            # Points artificiels pour les données manquantes
        }
        
        # Filtrer pour ne pas inclure les points artificiels
        real_points = self.df_combined[self.df_combined['IsArtificial'] == 0]
        
        # Recalculer les masques avec le nouveau seuil
        tp_mask = (real_points['DetectedAnomaly'] == 1) & (real_points['VraieAnomalie'] == 1)
        fp_mask = (real_points['DetectedAnomaly'] == 1) & (real_points['VraieAnomalie'] == 0)
        fn_mask = (real_points['DetectedAnomaly'] == 0) & (real_points['VraieAnomalie'] == 1)
        
        # Mettre à jour les données des scatter plots pour le premier graphique (valeurs)
        if self.scatter_tp_values:
            self.scatter_tp_values.set_offsets(np.column_stack([
                mdates.date2num(real_points.loc[tp_mask, 'Datetime']),
                real_points.loc[tp_mask, 'Valeur']
            ]))
            
        if self.scatter_fp_values:
            self.scatter_fp_values.set_offsets(np.column_stack([
                mdates.date2num(real_points.loc[fp_mask, 'Datetime']),
                real_points.loc[fp_mask, 'Valeur']
            ]))
            
        if self.scatter_fn_values:
            self.scatter_fn_values.set_offsets(np.column_stack([
                mdates.date2num(real_points.loc[fn_mask, 'Datetime']),
                real_points.loc[fn_mask, 'Valeur']
            ]))
            
        # Mettre à jour les données des scatter plots pour le deuxième graphique (scores)
        if self.scatter_tp_scores:
            self.scatter_tp_scores.set_offsets(np.column_stack([
                mdates.date2num(real_points.loc[tp_mask, 'Datetime']),
                real_points.loc[tp_mask, 'AnomalyScore']
            ]))
            
        if self.scatter_fp_scores:
            self.scatter_fp_scores.set_offsets(np.column_stack([
                mdates.date2num(real_points.loc[fp_mask, 'Datetime']),
                real_points.loc[fp_mask, 'AnomalyScore']
            ]))
            
        if self.scatter_fn_scores:
            self.scatter_fn_scores.set_offsets(np.column_stack([
                mdates.date2num(real_points.loc[fn_mask, 'Datetime']),
                real_points.loc[fn_mask, 'AnomalyScore']
            ]))
            
        # Mettre à jour la ligne de seuil
        if self.threshold_line:
                self.threshold_line.set_ydata([self.current_threshold, self.current_threshold])

    def update_confusion_matrix(self):
        """Met à jour la matrice de confusion et les métriques avec le nouveau seuil en incluant les points manquants"""
        if not hasattr(self, 'df_combined') or self.df_combined is None:
            return
            
        # Créer une copie du dataframe combiné pour les calculs
        df_eval = self.df_combined.copy()
        
        # Marquer tous les points artificiels (manquants) comme des vrais positifs
        # Ces points sont considérés comme des anomalies réelles (car ce sont des écarts temporels)
        # et ils ont été détectés comme tels par votre système
        df_eval.loc[df_eval['IsArtificial'] == 1, 'VraieAnomalie'] = 1
        df_eval.loc[df_eval['IsArtificial'] == 1, 'DetectedAnomaly'] = 1
        
        # Calculer la nouvelle matrice de confusion avec tous les points, y compris les artificiels
        cm = confusion_matrix(df_eval['VraieAnomalie'], df_eval['DetectedAnomaly'])
        
        # Mettre à jour le heatmap de la matrice de confusion
        if self.confusion_heatmap:
            self.ax_confusion.clear()
            self.confusion_heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Normal', 'Anomalie'], 
                            yticklabels=['Normal', 'Anomalie'], ax=self.ax_confusion)
            self.ax_confusion.set_title('Matrice de Confusion (incluant points manquants)')
            self.ax_confusion.set_xlabel('Prédit')
            self.ax_confusion.set_ylabel('Réel')
        
        # Calculer les nouvelles métriques
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        # Calculer le nombre de points artificiels (manquants) pour l'affichage
        num_artificial = sum(df_eval['IsArtificial'] == 1)
        
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Information sur le seuil optimal
        optimal_info = f'Oui, Score F1: {f1:.4f}' if (self.optimal_threshold is not None and abs(self.current_threshold - self.optimal_threshold) < 0.0001) else 'Non'
        
        # Créer un tableau mis à jour pour afficher les métriques
        metrics = [
            ['Précision (Precision)', f'{precision:.4f}'],
            ['Rappel (Recall)', f'{recall:.4f}'],
            ['Exactitude (Accuracy)', f'{accuracy:.4f}'],
            ['Score F1', f'{f1:.4f}'],
            ['Vrais Positifs (TP)', tp],
            ['Faux Positifs (FP)', fp],
            ['Vrais Négatifs (TN)', tn],
            ['Faux Négatifs (FN)', fn],
            ['Points manquants inclus', num_artificial],
            ['Total des points', total],
            ['Seuil utilisé', f'{self.current_threshold:.4f}'],
            ['Seuil optimal?', optimal_info]
        ]
        
        # Mettre à jour le tableau des métriques
        if self.metrics_table:
            self.ax_metrics.clear()
            self.ax_metrics.axis('tight')
            self.ax_metrics.axis('off')
            self.metrics_table = self.ax_metrics.table(cellText=metrics, colLabels=['Métrique', 'Valeur'], 
                            loc='center', cellLoc='center')
            self.metrics_table.auto_set_font_size(False)
            self.metrics_table.set_fontsize(12)
            self.metrics_table.scale(1.2, 1.5)
            
            # Mettre en évidence la ligne du seuil optimal si c'est le cas
            if self.optimal_threshold is not None and abs(self.current_threshold - self.optimal_threshold) < 0.0001:
                for key, cell in self.metrics_table._cells.items():
                    if key[0] == 12:  # Ligne "Seuil optimal?"
                        cell.set_facecolor('lightgreen')
                        cell.set_text_props(weight='bold')


    def on_threshold_change(self, val):
        """Callback pour gérer les changements de valeur du slider"""
        # Mettre à jour tous les graphiques avec le nouveau seuil
        self.update_plots(val)
    
    def find_optimal_threshold(self, metric='f1'):
        """
        Trouve le seuil optimal qui maximise une métrique donnée (f1, precision, recall) 
        en utilisant une approche par dichotomie
        
        Args:
            metric: La métrique à optimiser ('f1', 'precision', 'recall')
                    Par défaut: 'f1'
        
        Returns:
            Le seuil optimal trouvé
        """
        print(f"Recherche du seuil optimal pour maximiser: {metric}")
        
        # Définir la plage de recherche
        min_score = max(0, self.df_merged['AnomalyScore'].min() - 0.01)
        max_score = min(3, self.df_merged['AnomalyScore'].max() + 0.01)
        
        # Définir la précision de recherche
        precision = 0.0001
        
        # Définir la fonction d'évaluation selon la métrique choisie
        def evaluate_threshold(threshold):
            # Calculer les prédictions avec ce seuil
            y_pred = (self.df_merged['AnomalyScore'] > threshold).astype(int)
            y_true = self.df_merged['VraieAnomalie']
            
            # Calculer la métrique
            if metric == 'f1':
                # Le score F1 est la moyenne harmonique de la précision et du rappel
                return f1_score(y_true, y_pred)
            elif metric == 'precision':
                # Précision = VP / (VP + FP)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                return tp / (tp + fp) if (tp + fp) > 0 else 0
            elif metric == 'recall':
                # Rappel = VP / (VP + FN)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                return tp / (tp + fn) if (tp + fn) > 0 else 0
            else:
                raise ValueError(f"Métrique non reconnue: {metric}")
        
        # Recherche par dichotomie
        left = min_score
        right = max_score
        
        # Évaluer aux extrémités
        best_score = -1
        best_threshold = None
        
        # Division de l'espace de recherche
        steps = 0
        max_steps = 200  # Limiter le nombre d'itérations pour éviter les boucles infinies
        
        print(f"Plage de recherche initiale: [{left:.4f}, {right:.4f}]")
        
        while right - left > precision and steps < max_steps:
            steps += 1
            
            # Évaluer les points du tiers
            third = (right - left) / 3
            left_third = left + third
            right_third = right - third
            
            score_left = evaluate_threshold(left_third)
            score_right = evaluate_threshold(right_third)
            
            # Stocker le meilleur score trouvé
            if score_left > best_score:
                best_score = score_left
                best_threshold = left_third
            
            if score_right > best_score:
                best_score = score_right
                best_threshold = right_third
            
            # Réduire l'intervalle
            if score_left < score_right:
                left = left_third
            else:
                right = right_third
                
            # Debug
            print(f"Étape {steps}: Seuils [{left_third:.4f}, {right_third:.4f}], Scores [{score_left:.4f}, {score_right:.4f}]")
        
        # Recherche fine près du meilleur point trouvé
        # Tester avec une grille fine autour du point pour affiner
        fine_start = max(min_score, best_threshold - 0.01)
        fine_end = min(max_score, best_threshold + 0.01)
        fine_steps = 50
        fine_grid = np.linspace(fine_start, fine_end, fine_steps)
        
        for threshold in fine_grid:
            score = evaluate_threshold(threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"Seuil optimal trouvé: {best_threshold:.4f} avec {metric} = {best_score:.4f}")
        
        # Stocker le seuil optimal
        self.optimal_threshold = best_threshold
        
        return best_threshold
    
    def apply_optimal_threshold(self):
        """
        Trouve le seuil optimal et l'applique aux visualisations
        """
        if self.df_merged is None or len(self.df_merged) == 0:
            print("Aucune donnée pour trouver le seuil optimal")
            return
        
        # Si le seuil optimal n'a pas encore été calculé, le trouver
        if self.optimal_threshold is None:
            self.optimal_threshold = self.find_optimal_threshold(metric='f1')
        
        # Si un slider existe, mettre à jour sa valeur
        if self.threshold_slider is not None:
            self.threshold_slider.set_val(self.optimal_threshold)
        else:
            # Sinon, mettre à jour les graphiques directement
            self.update_plots(self.optimal_threshold)
        
        # Ajouter ou mettre à jour une ligne verticale dans le graphique des scores
        self.add_optimal_threshold_marker()
    
    def add_optimal_threshold_marker(self):
        """
        Ajoute ou met à jour un marqueur visuel pour le seuil optimal dans le graphique
        """
        if self.optimal_threshold is None or self.ax_scores is None:
            return
            
        # Supprimer l'ancienne ligne et annotation si elles existent
        if self.optimal_threshold_line is not None:
            self.optimal_threshold_line.remove()
        
        if self.optimal_threshold_annotation is not None:
            self.optimal_threshold_annotation.remove()
        
        # Ajouter une ligne verticale pour le seuil optimal avec une couleur différente
        y_min, y_max = self.ax_scores.get_ylim()
        self.optimal_threshold_line = self.ax_scores.axhline(
            y=self.optimal_threshold, 
            color='lime', 
            linestyle='-', 
            linewidth=2,
            alpha=0.7
        )
        
        # Ajouter une annotation
        # Trouver une position x appropriée (p. ex. 10% depuis la gauche du graphique)
        x_range = self.ax_scores.get_xlim()
        annotation_x = mdates.num2date(x_range[0] + 0.1 * (x_range[1] - x_range[0]))
        
        self.optimal_threshold_annotation = self.ax_scores.annotate(
            f'Seuil optimal: {self.optimal_threshold:.4f}',
            xy=(annotation_x, self.optimal_threshold),
            xytext=(annotation_x, self.optimal_threshold + 0.1 * (y_max - y_min)),
            arrowprops=dict(facecolor='lime', shrink=0.05, width=2, headwidth=8),
            color='lime',
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lime", alpha=0.8)
        )
        
        # Redessiner le graphique
        if self.fig_results:
            self.fig_results.canvas.draw_idle()
    
    def plot_results_interactive(self):
        """Crée la visualisation interactive des résultats avec un slider pour le seuil"""
        # D'abord, détecter et ajouter les points manquants
        self.detect_and_add_missing_points()
        
        if self.df_combined is None or len(self.df_combined) == 0:
            print("Aucune donnée à visualiser")
            return
        
        # Créer une figure avec deux subplots et un espace pour le slider
        self.fig_results, (self.ax_values, self.ax_scores) = plt.subplots(2, 1, figsize=(18, 14), 
                                                                         sharex=True,
                                                                         gridspec_kw={'height_ratios': [3, 2],
                                                                                     'bottom': 0.2})  # Espace pour le slider
        self.fig_results.suptitle('Visualisation des Résultats de Détection d\'Anomalies', fontsize=16)
        
        # Définir les couleurs pour les différents types de points
        colors = {
            'true_positive': 'red',           # Détecté comme anomalie + Est réellement une anomalie
            'false_positive': 'orange',       # Détecté comme anomalie + N'est pas une anomalie
            'true_negative': 'green',         # Non détecté comme anomalie + N'est pas une anomalie
            'false_negative': 'magenta',      # Non détecté comme anomalie + Est une anomalie
            'artificial': 'purple'            # Points artificiels pour les données manquantes
        }
        
        # --- PREMIER GRAPHIQUE: VALEURS ET PRÉDICTIONS ---
        # Tracer les valeurs originales et les prédictions
        self.ax_values.plot(self.df_combined['Datetime'], self.df_combined['Valeur'], 'b-', 
                          label='Valeur Originale', linewidth=1.0)
        self.ax_values.plot(self.df_combined['Datetime'], self.df_combined['Prediction'], 'g--', 
                          label='Prédiction', linewidth=1.0)
        
        # Filtrer pour ne pas inclure les points artificiels
        real_points = self.df_combined[self.df_combined['IsArtificial'] == 0]
        
        # Vrais positifs: détection correcte d'anomalie
        tp_mask = (real_points['DetectedAnomaly'] == 1) & (real_points['VraieAnomalie'] == 1)
        self.scatter_tp_values = self.ax_values.scatter(real_points.loc[tp_mask, 'Datetime'], 
                                                     real_points.loc[tp_mask, 'Valeur'], 
                                                     color=colors['true_positive'], s=80, marker='o', 
                                                     label='Vraie Anomalie (Correctement Détectée)')
        
        # Faux positifs: fausse alerte
        fp_mask = (real_points['DetectedAnomaly'] == 1) & (real_points['VraieAnomalie'] == 0)
        self.scatter_fp_values = self.ax_values.scatter(real_points.loc[fp_mask, 'Datetime'], 
                                                     real_points.loc[fp_mask, 'Valeur'], 
                                                     color=colors['false_positive'], s=80, marker='o', 
                                                     label='Fausse Alerte')
                
        # Faux négatifs: anomalie manquée
        fn_mask = (real_points['DetectedAnomaly'] == 0) & (real_points['VraieAnomalie'] == 1)
        self.scatter_fn_values = self.ax_values.scatter(real_points.loc[fn_mask, 'Datetime'], 
                                                     real_points.loc[fn_mask, 'Valeur'], 
                                                     color=colors['false_negative'], s=80, marker='X', 
                                                     label='Anomalie Manquée')
        
        # Points artificiels pour les écarts temporels
        artificial_mask = self.df_combined['IsArtificial'] == 1
        if artificial_mask.any():
            self.ax_values.scatter(self.df_combined.loc[artificial_mask, 'Datetime'], 
                                 self.df_combined.loc[artificial_mask, 'Valeur'], 
                                 color=colors['artificial'], s=60, marker='s', 
                                 label='Points Manquants')
            
            # Ajouter des annotations pour les séquences de points manquants
            artificial_indices = np.where(artificial_mask)[0]
            if len(artificial_indices) > 0:
                # Identifier les débuts de séquences
                sequence_starts = [artificial_indices[0]]
                for i in range(1, len(artificial_indices)):
                    if artificial_indices[i] > artificial_indices[i-1] + 1:
                        sequence_starts.append(artificial_indices[i])
                
                # Ajouter des annotations pour chaque séquence
                for start_idx in sequence_starts:
                    # Trouver la fin de la séquence
                    end_idx = start_idx
                    while end_idx + 1 < len(self.df_combined) and self.df_combined['IsArtificial'].iloc[end_idx + 1] == 1:
                        end_idx += 1
                    
                    # Nombre de points dans la séquence
                    num_points = end_idx - start_idx + 1
                    if num_points > 1:
                        # Position de l'annotation (milieu de la séquence)
                        mid_idx = (start_idx + end_idx) // 2
                        mid_time = self.df_combined['Datetime'].iloc[mid_idx]
                        mid_value = self.df_combined['Valeur'].iloc[mid_idx]
                        
                        # Ajouter l'annotation
                        self.ax_values.annotate(f"{num_points} points manquants", 
                                              xy=(mid_time, mid_value), 
                                              xytext=(0, 30), 
                                              textcoords='offset points',
                                              arrowprops=dict(arrowstyle='->', color='purple'),
                                              color='purple',
                                              fontsize=10,
                                              ha='center')
        
        # Configuration du premier graphique
        self.ax_values.set_title('Comparaison des Valeurs Réelles et Prédites')
        self.ax_values.set_ylabel('Valeur')
        self.ax_values.grid(True)
        self.ax_values.legend(loc='upper right')
        
        # --- DEUXIÈME GRAPHIQUE: SCORES D'ANOMALIE ET SEUIL ---
        # Tracer les scores d'anomalie
        self.ax_scores.plot(self.df_combined['Datetime'], self.df_combined['AnomalyScore'], 'b-', 
                          label='Score d\'Anomalie', linewidth=1.0)
        
        # Tracer le seuil d'anomalie initial
        self.threshold_line = self.ax_scores.axhline(y=self.current_threshold, color='r', linestyle='--', 
                                                   label=f'Seuil d\'Anomalie ({self.current_threshold:.4f})')
        
        # Ajouter des points pour les différents résultats sur le graphique de score
        self.scatter_tp_scores = self.ax_scores.scatter(real_points.loc[tp_mask, 'Datetime'], 
                                                     real_points.loc[tp_mask, 'AnomalyScore'], 
                                                     color=colors['true_positive'], s=80, marker='o')
                
        self.scatter_fp_scores = self.ax_scores.scatter(real_points.loc[fp_mask, 'Datetime'], 
                                                     real_points.loc[fp_mask, 'AnomalyScore'], 
                                                     color=colors['false_positive'], s=80, marker='o')
                
        self.scatter_fn_scores = self.ax_scores.scatter(real_points.loc[fn_mask, 'Datetime'], 
                                                     real_points.loc[fn_mask, 'AnomalyScore'], 
                                                     color=colors['false_negative'], s=80, marker='X')
        
        # Configuration du deuxième graphique
        self.ax_scores.set_title('Scores d\'Anomalie et Seuil')
        self.ax_scores.set_xlabel('Temps')
        self.ax_scores.set_ylabel('Score')
        self.ax_scores.grid(True)
        self.ax_scores.legend(loc='upper right')
        
        # Configurer l'axe x pour afficher les dates correctement
        self.ax_scores.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M:%S'))
        self.fig_results.autofmt_xdate()  # Rotation des étiquettes de date
        
        # --- AJOUTER UN SLIDER POUR LE SEUIL ---
        # Déterminer les valeurs min et max pour le slider
        min_score = self.df_merged['AnomalyScore'].min()
        max_score = self.df_merged['AnomalyScore'].max()
        
        # Ajuster les limites du slider pour avoir une marge
        slider_min = max(0, min_score - 0.05)
        slider_max = min(1, max_score + 0.05)
        
        # Créer un axe séparé pour le slider
        ax_slider = self.fig_results.add_axes([0.2, 0.07, 0.65, 0.03])  # [left, bottom, width, height]
        
        # Créer le slider
        self.threshold_slider = Slider(
            ax=ax_slider,
            label='Seuil d\'Anomalie',
            valmin=slider_min,
            valmax=slider_max,
            valinit=self.current_threshold,
            valstep=0.001
        )
        
        # Ajouter un axe pour le bouton d'optimisation
        ax_button = self.fig_results.add_axes([0.8, 0.01, 0.15, 0.04])  # [left, bottom, width, height]
        optimize_button = Button(ax_button, 'Trouver Seuil Optimal', color='lightgreen', hovercolor='palegreen')
        optimize_button.on_clicked(lambda event: self.apply_optimal_threshold())
        
        # Ajouter un texte pour indiquer la valeur actuelle du seuil
        threshold_text = self.fig_results.text(0.5, 0.03, f'Seuil actuel: {self.current_threshold:.4f}', 
                                             ha='center', va='center')
        
        # Fonction de callback pour mettre à jour le texte du seuil
        def update_threshold_text(val):
            threshold_text.set_text(f'Seuil actuel: {val:.4f}')
            
        # Connecter les fonctions de callback au slider
        self.threshold_slider.on_changed(self.on_threshold_change)
        self.threshold_slider.on_changed(update_threshold_text)
        
        # Ajuster la mise en page
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.15)  # Laisser de l'espace pour le slider
        
        # Créer également la fenêtre de matrice de confusion
        self.plot_confusion_matrix_interactive()
        
        # Si on a déjà calculé un seuil optimal, le marquer sur le graphique
        if self.optimal_threshold is not None:
            self.add_optimal_threshold_marker()
        
        # Afficher les figures
        plt.show()
    
    def plot_confusion_matrix_interactive(self):
        """Affiche la matrice de confusion et les métriques de performance de manière interactive"""
        if self.df_merged is None or len(self.df_merged) == 0:
            print("Aucune donnée pour calculer la matrice de confusion")
            return
            
        # Calculer la matrice de confusion initiale
        cm = confusion_matrix(self.df_merged['VraieAnomalie'], self.df_merged['DetectedAnomaly'])
        
        # Créer une figure
        self.fig_confusion, (self.ax_confusion, self.ax_metrics) = plt.subplots(1, 2, figsize=(18, 8), 
                                                                             gridspec_kw={'width_ratios': [1, 2]})
        
        # Afficher la matrice de confusion avec seaborn
        self.confusion_heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                          xticklabels=['Normal', 'Anomalie'], 
                                          yticklabels=['Normal', 'Anomalie'], ax=self.ax_confusion)
        self.ax_confusion.set_title('Matrice de Confusion')
        self.ax_confusion.set_xlabel('Prédit')
        self.ax_confusion.set_ylabel('Réel')
        
        # Calculer les métriques initiales
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Créer un tableau pour afficher les métriques
        metrics = [
            ['Précision (Precision)', f'{precision:.4f}'],
            ['Rappel (Recall)', f'{recall:.4f}'],
            ['Exactitude (Accuracy)', f'{accuracy:.4f}'],
            ['Score F1', f'{f1:.4f}'],
            ['Vrais Positifs (TP)', tp],
            ['Faux Positifs (FP)', fp],
            ['Vrais Négatifs (TN)', tn],
            ['Faux Négatifs (FN)', fn],
            ['Total des points', total],
            ['Seuil utilisé', f'{self.current_threshold:.4f}'],
            ['Seuil optimal?', 'Non']
        ]
        
        # Afficher le tableau des métriques
        self.ax_metrics.axis('tight')
        self.ax_metrics.axis('off')
        self.metrics_table = self.ax_metrics.table(cellText=metrics, colLabels=['Métrique', 'Valeur'], 
                                                loc='center', cellLoc='center')
        self.metrics_table.auto_set_font_size(False)
        self.metrics_table.set_fontsize(12)
        self.metrics_table.scale(1.2, 1.5)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Afficher également le rapport de classification
        print("\nRapport de classification initial:")
        print(classification_report(self.df_merged['VraieAnomalie'], self.df_merged['DetectedAnomaly'],
                                  target_names=['Normal', 'Anomalie']))
                                  
    def run_analysis(self):
        """Exécute l'analyse interactive avec slider"""
        self.plot_results_interactive()
        # La matrice de confusion est déjà appelée depuis plot_results_interactive

if __name__ == "__main__":
    # Configurer les arguments de ligne de commande
    parser = argparse.ArgumentParser(description='Visualiseur interactif de résultats de détection d\'anomalies')
    parser.add_argument('--original', type=str, required=True, 
                        help='Chemin vers le fichier CSV original (Date;Heure;Valeur;Flag)')
    parser.add_argument('--analysis', type=str, required=True, 
                        help='Chemin vers le fichier CSV d\'analyse (Date;Heure;ValeurOriginale;...)')
    parser.add_argument('--start', type=str, 
                        help='Point de départ (index numérique ou date au format DD/MM/YYYY HH:MM:SS)')
    
    # Analyser les arguments
    try:
        args = parser.parse_args()
        original_path = args.original
        analysis_path = args.analysis
        start_point = None
        
        # Convertir start_point en int si c'est un nombre, sinon garder la chaîne
        if args.start:
            try:
                start_point = int(args.start)
            except ValueError:
                # Si ce n'est pas un entier, considérer comme une date
                start_point = args.start
    except:
        # Code pour les environnements sans ligne de commande...
        original_path = "../data/test_1.3.csv"
        analysis_path = "../result/test_1.3_anomalies.csv"
        start_point = 00  # Ou définir une valeur par défaut

    try:
        # Créer le visualiseur avec le nouveau paramètre
        visualizer = AnomalyResultVisualizer(original_path, analysis_path, start_point)
        visualizer.run_analysis()
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")        
        # Afficher un message d'aide
        print("\nUtilisation: python anomaly_visualizer.py --original [chemin_original] --analysis [chemin_analyse]")
        print("\nLe fichier original doit contenir les colonnes: Date;Heure;Valeur;Flag")
        print("Le fichier d'analyse doit contenir les colonnes: Date;Heure;ValeurOriginale;Prediction;AnomalyScore;SeuilAnomalie;EstAnomalie")
