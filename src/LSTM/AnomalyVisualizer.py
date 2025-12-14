import os
# Définir le backend avant d'importer matplotlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl

class AnomalyVisualizer:
    def __init__(self, detector):
        """Initialise le visualiseur d'anomalies"""
        self.detector = detector
        
        # Désactiver l'échantillonnage automatique de matplotlib
        mpl.rcParams['path.simplify'] = False
        mpl.rcParams['path.simplify_threshold'] = 1.0
        mpl.rcParams['agg.path.chunksize'] = 10000
        
        # Configuration pour forcer le rendu
        mpl.rcParams['figure.autolayout'] = True
        plt.ion()  # Activer le mode interactif dès le début
        
        # Listes pour stocker les anomalies détectées
        self.anomalies = {
            'severe': {'times': [], 'values': [], 'errors': []},
            'moderate': {'times': [], 'values': [], 'errors': []}
        }
        
        # Compteur de points
        self.point_counter = 0
        
        # La figure est initialement None, sera créée lors de la première actualisation
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.clean_values_line = None

    def create_new_figure(self):
        """Crée une nouvelle figure pour l'affichage"""
        # Forcer la fermeture de la figure précédente
        if self.fig is not None:
            plt.figure(self.fig.number)  # Activer la figure
            plt.close()  # Fermer la figure active
        
        # Créer une nouvelle figure
        plt.ion()  # Assurer que le mode interactif est activé
        self.fig = plt.figure(figsize=(18, 12), dpi=100)
        
        # Créer les subplots manuellement pour plus de contrôle
        gs = self.fig.add_gridspec(2, 1)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        
        # Graphique principal - signal
        self.observed_line, = self.ax1.plot([], [], 'b-', label="Signal observé (brut)", linewidth=1.0)
        self.clean_values_line, = self.ax1.plot([], [], 'g-', label="Signal nettoyé", linewidth=1.0)
        self.predicted_line, = self.ax1.plot([], [], 'r--', label="Reconstruction", linewidth=1.0)
        
        self.ax1.set_xlabel("Temps")
        self.ax1.set_ylabel("Valeur")
        self.ax1.xaxis_date()
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M:%S'))
        self.ax1.legend()
        self.ax1.set_title(f"Détection d'anomalies - {self.point_counter} points traités")
        self.ax1.grid(True)
        
        # Graphique inférieur - erreur
        self.error_line, = self.ax2.plot([], [], 'g-', label="Erreur de reconstruction", linewidth=1.0)
        self.threshold_line, = self.ax2.plot([], [], 'r--', label="Seuil d'anomalie", linewidth=1.0)
        self.ax2.set_xlabel("Temps")
        self.ax2.set_ylabel("Erreur")
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Préparation des marqueurs d'anomalies
        self.severe_anomalies = self.ax1.scatter([], [], color='magenta', marker='o', s=80, label="Anomalie sévère")
        self.moderate_anomalies = self.ax1.scatter([], [], color='orange', marker='o', s=60, label="Anomalie modérée")
        
        # Même chose pour le graphique d'erreur
        self.severe_error_marks = self.ax2.scatter([], [], color='magenta', marker='o', s=80)
        self.moderate_error_marks = self.ax2.scatter([], [], color='orange', marker='o', s=60)
        
        # Ajustement de la mise en page
        self.fig.tight_layout()
        
        # Forcer le rendu initial
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def update_plot(self):
        """Met à jour la visualisation tous les 100 points"""
        # Incrémenter le compteur
        self.point_counter += 1
        
        # Si ce n'est pas un multiple de 100, ne rien faire
        if self.point_counter % 2000 != 0 and self.point_counter != 1:
            return False
        
        # Obtenir les données depuis le détecteur
        data = self.detector.get_visual_data()
        
        # Vérifier s'il y a des données à afficher
        if not data['is_trained'] or not data['values']:
            return False
        
        # Créer une nouvelle figure et ferme l'ancienne
        self.create_new_figure()
        
        # Maintenant que la figure est créée, mettre à jour les lignes
        times = data['times']
        values = data['values']
        clean_values = data.get('clean_values', values)  # Utiliser les valeurs brutes si pas de nettoyées
        
        # Conversion explicite des dates en valeurs numériques pour le traçage
        time_nums = mdates.date2num(times)
        self.observed_line.set_data(time_nums, values)
        
        # Ajouter la ligne des valeurs nettoyées si disponible
        if hasattr(self, 'clean_values_line') and self.clean_values_line is not None:
            self.clean_values_line.set_data(time_nums, clean_values)
        
        # Vérifier s'il y a des données à afficher
        if not data['is_trained'] or not data['values']:
            return False
        
        # Créer une nouvelle figure et ferme l'ancienne
        self.create_new_figure()
        
        # Mise à jour des lignes principales avec TOUS les points depuis le début
        times = data['times']
        values = data['values']
        
        # Conversion explicite des dates en valeurs numériques pour le traçage
        time_nums = mdates.date2num(times)
        self.observed_line.set_data(time_nums, values)
        
        if data['predictions']:
            # Afficher toutes les prédictions disponibles
            pred_times = times[-len(data['predictions']):]
            pred_time_nums = mdates.date2num(pred_times)
            self.predicted_line.set_data(pred_time_nums, data['predictions'])
        
        if data['errors']:
            # Afficher toutes les erreurs disponibles
            error_times = times[-len(data['errors']):]
            error_time_nums = mdates.date2num(error_times)
            self.error_line.set_data(error_time_nums, data['errors'])
            
            # Seuil d'anomalie
            if data['threshold'] is not None:
                time_min = time_nums[0] if len(time_nums) > 0 else 0
                time_max = time_nums[-1] if len(time_nums) > 0 else 1
                self.threshold_line.set_data([time_min, time_max], [data['threshold'], data['threshold']])
        
        # Mise à jour des marqueurs d'anomalies
        if self.anomalies['severe']['times']:
            severe_t = self.anomalies['severe']['times']
            severe_v = self.anomalies['severe']['values']
            severe_e = self.anomalies['severe']['errors']
            
            self.severe_anomalies.set_offsets([(mdates.date2num(t), v) for t, v in zip(severe_t, severe_v)])
            self.severe_error_marks.set_offsets([(mdates.date2num(t), e) for t, e in zip(severe_t, severe_e)])
        else:
            self.severe_anomalies.set_offsets(np.empty((0, 2)))
            self.severe_error_marks.set_offsets(np.empty((0, 2)))
            
        if self.anomalies['moderate']['times']:
            moderate_t = self.anomalies['moderate']['times']
            moderate_v = self.anomalies['moderate']['values']
            moderate_e = self.anomalies['moderate']['errors']
            
            self.moderate_anomalies.set_offsets([(mdates.date2num(t), v) for t, v in zip(moderate_t, moderate_v)])
            self.moderate_error_marks.set_offsets([(mdates.date2num(t), e) for t, e in zip(moderate_t, moderate_e)])
        else:
            self.moderate_anomalies.set_offsets(np.empty((0, 2)))
            self.moderate_error_marks.set_offsets(np.empty((0, 2)))
        
        # Ajustement des limites pour afficher TOUS les points
                # Ajustement des limites pour afficher TOUS les points sans zoom
        if len(time_nums) > 0:
            # Forcer l'affichage de tous les points sans zoom
            self.ax1.set_xlim(time_nums[0], time_nums[-1])
            self.ax2.set_xlim(time_nums[0], time_nums[-1])
            
            # Réinitialiser tout zoom précédent
            self.ax1.autoscale(True, 'both', tight=False)
            self.ax2.autoscale(True, 'both', tight=False)
            
            # Désactiver l'autoscaling après avoir défini les limites
            self.ax1.autoscale(False)
            self.ax2.autoscale(False)
            
            # Ajuster l'échelle y pour les valeurs explicitement
            y_min = min(values) - 0.1 * (max(values) - min(values)) if len(values) > 1 else -1
            y_max = max(values) + 0.1 * (max(values) - min(values)) if len(values) > 1 else 1
            self.ax1.set_ylim(y_min, y_max)
            
            # Ajuster l'échelle y pour les erreurs explicitement
            if data['errors']:
                max_error = max(max(data['errors']), data['threshold'] * 1.5 if data['threshold'] else 0)
                self.ax2.set_ylim(0, max_error * 1.2)
        # Forces de multiples manières le rafraîchissement complet
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Pause pour s'assurer que le rendu est complet (crucial!)
        # plt.pause(5)  # Pause plus longue pour garantir le rendu
        
        print(f"Mise à jour de la visualisation: {self.point_counter} points traités")
        
        return True
        
    def add_anomaly(self, time, value, error, severe=False):
        """Ajoute une anomalie détectée à la visualisation"""
        anomaly_type = 'severe' if severe else 'moderate'
        
        self.anomalies[anomaly_type]['times'].append(time)
        self.anomalies[anomaly_type]['values'].append(value)
        self.anomalies[anomaly_type]['errors'].append(error)
        
    def clear_anomalies(self):
        """Effacer toutes les anomalies enregistrées"""
        for anomaly_type in self.anomalies:
            self.anomalies[anomaly_type]['times'] = []
            self.anomalies[anomaly_type]['values'] = []
            self.anomalies[anomaly_type]['errors'] = []
    
    def show(self):
        """Affiche la figure (bloquant)"""
        plt.ioff()
        if self.fig is not None:
            plt.figure(self.fig.number)
            plt.show(block=True)


