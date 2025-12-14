import os
# Définir le backend avant d'importer matplotlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from DetectPeriodRobust import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button, CheckButtons, Slider


class PeriodSelectorWithZoom:
    def __init__(self, data, times, periods_info):
        self.data = data
        self.times = pd.to_datetime(times) if not isinstance(times[0], pd.Timestamp) else times
        self.periods_info = periods_info
        self.selected_periods = []
        self.start_idx = 0
        
        # Pour le zoom et la navigation
        self.orig_xlim = None
        self.orig_ylim = None
        
        # Création de la figure principale
        plt.ioff()
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle("Sélection de périodes avec zoom", fontsize=16)
        
        # Subplot du signal
        self.ax_signal = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
        self.ax_signal.plot(self.times, self.data, 'b-')
        self.ax_signal.set_xlabel('Temps')
        self.ax_signal.set_ylabel('Valeur')
        self.ax_signal.grid(True)
        self.ax_signal.set_title('Signal temporel avec périodes détectées')
        self.ax_signal.xaxis_date()
        self.ax_signal.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
        
        # Subplot de l'autocorrélation
        max_lag = min(len(data) // 2, 1000)
        data_norm = (data - np.mean(data)) / (np.std(data) if np.std(data) > 0 else 1)
        autocorr = np.correlate(data_norm, data_norm, mode='full')
        self.autocorr = autocorr[len(data)-1:len(data)-1+max_lag] / len(data)
        
        self.ax_acf = plt.subplot2grid((5, 1), (3, 0), rowspan=1)
        self.ax_acf.plot(range(len(self.autocorr)), self.autocorr)
        self.ax_acf.set_title('Autocorrélation')
        self.ax_acf.set_xlabel('Lag (points)')
        self.ax_acf.set_ylabel('Autocorrélation')
        self.ax_acf.grid(True)
        
        # Subplot du spectre (FFT)
        data_detrended = data - np.mean(data)
        yf = fft(data_detrended)
        n = len(data)
        xf = fftfreq(n, 1)[:n//2]
        self.magnitude = 2.0/n * np.abs(yf[:n//2])
        
        self.ax_fft = plt.subplot2grid((5, 1), (4, 0), rowspan=1)
        
        # Pour la FFT, on trace en fonction de la période plutôt que de la fréquence
        self.periods_x = np.zeros_like(xf)
        for i, freq in enumerate(xf):
            self.periods_x[i] = 1 / freq if freq > 0 else 0
        
        # Filtrer les périodes trop grandes pour l'affichage
        valid_mask = (self.periods_x > 0) & (self.periods_x <= 1000)
        self.ax_fft.plot(self.periods_x[valid_mask], self.magnitude[valid_mask])
        self.ax_fft.set_title('Spectre de fréquence (FFT)')
        self.ax_fft.set_xlabel('Période (points)')
        self.ax_fft.set_ylabel('Amplitude')
        self.ax_fft.grid(True)
        
        # Ajout des marqueurs pour les périodes détectées
        self.period_markers = []
        self.period_texts = []
        self.period_spans = []
        
        # Tracer les périodes détectées sur les graphiques
        colors = {'ACF': 'r', 'FFT': 'g', 'VALIDATED': 'm', 'DEFAULT': 'k'}
        for i, (period, score, method) in enumerate(self.periods_info):
            color = colors.get(method, 'b')
            
            # Marquer sur l'autocorrélation
            if period < len(self.autocorr):
                marker = self.ax_acf.axvline(x=period, color=color, linestyle='--', alpha=0.7)
                self.period_markers.append(marker)
                text = self.ax_acf.text(period, self.autocorr[period] if period < len(self.autocorr) else 0, 
                                        f"P{i+1}={period}", color=color, ha='center', va='bottom',
                                        bbox=dict(facecolor='white', alpha=0.5))
                self.period_texts.append(text)
            
            # Marquer sur le spectre FFT
            period_idx = np.argmin(np.abs(self.periods_x - period))
            if period_idx < len(self.magnitude) and valid_mask[period_idx]:
                marker = self.ax_fft.axvline(x=period, color=color, linestyle='--', alpha=0.7)
                self.period_markers.append(marker)
                text = self.ax_fft.text(period, self.magnitude[period_idx], 
                                        f"P{i+1}={period}", color=color, ha='center', va='bottom',
                                        bbox=dict(facecolor='white', alpha=0.5))
                self.period_texts.append(text)
            
            # Exemple de motif sur le signal principal (pour la première occurrence)
            if len(self.data) > period + self.start_idx:
                # Dessiner un rectangle pour montrer un exemple de la période
                x_start = self.times[self.start_idx]
                x_end = self.times[self.start_idx + period]
                y_min, y_max = self.ax_signal.get_ylim()
                span = self.ax_signal.axvspan(x_start, x_end, alpha=0.2, color=color)
                self.period_spans.append(span)
                
                # Étiqueter la période
                mid_x = x_start + (x_end - x_start) / 2
                label = self.ax_signal.text(mid_x, y_max - (y_max - y_min) * 0.05, 
                                     f"P{i+1}={period}", color='black', 
                                     ha='center', va='top',
                                     bbox=dict(facecolor=color, alpha=0.5))
                self.period_texts.append(label)
        
        # Ajout d'une liste à cocher pour sélectionner les périodes
        ax_check = plt.axes([0.05, 0.05, 0.2, 0.1])
        period_labels = [f"P{i+1}: {p} points" for i, (p, _, _) in enumerate(self.periods_info)]
        self.check = CheckButtons(ax_check, period_labels, [False] * len(period_labels))
        self.check.on_clicked(self.check_callback)
        
        # Bouton de validation
        ax_button = plt.axes([0.8, 0.05, 0.15, 0.075])
        self.btn = Button(ax_button, 'Valider')
        self.btn.on_clicked(self.validate)
        
        # Sliders pour ajuster l'indice de démarrage
        ax_slider = plt.axes([0.3, 0.05, 0.4, 0.05])
        self.slider = Slider(ax_slider, 'Point de départ', 0, len(data)-1, valinit=0, valstep=1)
        self.slider.on_changed(self.update_start_idx)
        
        # Instructions pour le zoom
        self.fig.text(0.5, 0.01, 
                   "Zoom: molette de souris ou touche +/-, déplacement: touches flèches, réinitialiser zoom: touche r", 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(facecolor='lightblue', alpha=0.5))
        
        # Connecter les événements clavier et souris
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Enregistrer les limites originales pour pouvoir réinitialiser le zoom
        self.orig_xlim = self.ax_signal.get_xlim()
        self.orig_ylim = self.ax_signal.get_ylim()
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Ajuster pour laisser de la place aux contrôles
        
    def check_callback(self, label):
        # Analyse le label pour récupérer l'indice et la période
        for i, per_label in enumerate([f"P{j+1}: {p} points" for j, (p, _, _) in enumerate(self.periods_info)]):
            if label == per_label:
                period = self.periods_info[i][0]
                if period in self.selected_periods:
                    self.selected_periods.remove(period)
                else:
                    self.selected_periods.append(period)
                # Mettre à jour l'affichage de sélection
                self.update_selection_display()
                break
    
    def update_selection_display(self):
        # Afficher les périodes sélectionnées
        title = f"Signal temporel - Périodes sélectionnées: {self.selected_periods}"
        self.ax_signal.set_title(title)
        self.fig.canvas.draw_idle()
    
    def update_start_idx(self, val):
        # Mettre à jour l'indice de départ
        self.start_idx = int(val)
        
        # Mettre à jour les zones de motif sur le signal
        for span in self.period_spans:
            span.remove()
        self.period_spans = []
        
        # Mise à jour des exemples de motifs
        colors = {'ACF': 'r', 'FFT': 'g', 'VALIDATED': 'm', 'DEFAULT': 'k'}
        for i, (period, _, method) in enumerate(self.periods_info):
            color = colors.get(method, 'b')
            if len(self.data) > period + self.start_idx:
                # Dessiner un rectangle pour montrer un exemple de la période
                x_start = self.times[self.start_idx]
                x_end = self.times[self.start_idx + period]
                span = self.ax_signal.axvspan(x_start, x_end, alpha=0.2, color=color)
                self.period_spans.append(span)
                
                # Étiquette de période
                y_min, y_max = self.ax_signal.get_ylim()
                mid_x = x_start + (x_end - x_start) / 2
                label = self.ax_signal.text(mid_x, y_max - (y_max - y_min) * 0.05, 
                                     f"P{i+1}={period}", color='black', 
                                     ha='center', va='top',
                                     bbox=dict(facecolor=color, alpha=0.5))
                self.period_texts.append(label)
        
        self.fig.canvas.draw_idle()
    
    def on_scroll(self, event):
        """Gère le zoom avec la molette de souris"""
        # Vérifier sur quel axe le zoom doit s'appliquer
        if event.inaxes == self.ax_signal:
            ax = self.ax_signal
        elif event.inaxes == self.ax_acf:
            ax = self.ax_acf
        elif event.inaxes == self.ax_fft:
            ax = self.ax_fft
        else:
            return
        
        factor = 1.2 if event.button == 'up' else 1/1.2
        
        # Limites actuelles
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        x = event.xdata
        y = event.ydata
        
        new_width = (xlim[1] - xlim[0]) / factor
        new_height = (ylim[1] - ylim[0]) / factor
        
        x_ratio = (x - xlim[0]) / (xlim[1] - xlim[0])
        y_ratio = (y - ylim[0]) / (ylim[1] - ylim[0])
        
        new_xlim = [x - x_ratio * new_width, x + (1-x_ratio) * new_width]
        new_ylim = [y - y_ratio * new_height, y + (1-y_ratio) * new_height]
        
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.fig.canvas.draw_idle()
    
    def on_key(self, event):
        """Gère les événements clavier pour le zoom et le déplacement"""
        # Déterminer quel axe est actif
        if event.inaxes == self.ax_signal:
            ax = self.ax_signal
        elif event.inaxes == self.ax_acf:
            ax = self.ax_acf
        elif event.inaxes == self.ax_fft:
            ax = self.ax_fft
        else:
            ax = self.ax_signal
        
        if event.key == '+':
            # Zoom in
            self.zoom(ax, 1.2)
        elif event.key == '-':
            # Zoom out
            self.zoom(ax, 1/1.2)
        elif event.key == 'left':
            # Déplacer vers la gauche
            self.pan(ax, -0.1, 0)
        elif event.key == 'right':
            # Déplacer vers la droite
            self.pan(ax, 0.1, 0)
        elif event.key == 'up':
            # Déplacer vers le haut
            self.pan(ax, 0, 0.1)
        elif event.key == 'down':
            # Déplacer vers le bas
            self.pan(ax, 0, -0.1)
        elif event.key == 'r':
            # Réinitialiser le zoom
            self.reset_zoom(ax)
    
    def zoom(self, ax, factor):
        """Fonction de zoom centrée sur le milieu de la vue actuelle"""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2
        
        new_width = (xlim[1] - xlim[0]) / factor
        new_height = (ylim[1] - ylim[0]) / factor
        
        new_xlim = [xmid - new_width/2, xmid + new_width/2]
        new_ylim = [ymid - new_height/2, ymid + new_height/2]
        
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.fig.canvas.draw_idle()
    
    def pan(self, ax, dx, dy):
        """Fonction de déplacement relatif en pourcentage de la vue actuelle"""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        x_shift = x_range * dx
        y_shift = y_range * dy
        
        new_xlim = [xlim[0] + x_shift, xlim[1] + x_shift]
        new_ylim = [ylim[0] + y_shift, ylim[1] + y_shift]
        
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.fig.canvas.draw_idle()
    
    def reset_zoom(self, ax):
        """Réinitialise le zoom à la vue d'origine"""
        if ax == self.ax_signal and self.orig_xlim is not None and self.orig_ylim is not None:
            ax.set_xlim(self.orig_xlim)
            ax.set_ylim(self.orig_ylim)
        elif ax == self.ax_acf:
            ax.set_xlim(0, len(self.autocorr))
            ax.set_ylim(min(self.autocorr), max(self.autocorr))
        elif ax == self.ax_fft:
            valid_mask = (self.periods_x > 0) & (self.periods_x <= 1000)
            ax.set_xlim(min(self.periods_x[valid_mask]), max(self.periods_x[valid_mask]))
            ax.set_ylim(0, max(self.magnitude[valid_mask]) * 1.1)
        
        self.fig.canvas.draw_idle()
    
    def validate(self, event):
        """Termine la sélection et ferme la figure"""
        plt.close(self.fig)
    
    def show(self):
        """Affiche l'interface et retourne les périodes sélectionnées"""
        plt.ion()
        plt.show(block=True)
        
        if not self.selected_periods:
            self.selected_periods = [p for p, _, _ in self.periods_info[:min(3, len(self.periods_info))]]
            if not self.selected_periods:
                self.selected_periods = [100]
        
        return self.selected_periods, self.start_idx
    


