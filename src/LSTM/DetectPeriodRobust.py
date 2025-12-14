import os
# Définir le backend avant d'importer matplotlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter

# ------------------------------------------------
# Fonctions pour la détection de périodes (réutilisées)
# ------------------------------------------------
def detect_periods_robust(data, min_period=10, max_period=None, 
                          acf_threshold=0.3, fft_threshold=0.3, max_detections=10):
    """
    Détecte les périodes significatives dans les données en utilisant différentes méthodes.
    
    Args:
        data: Données à analyser (array numpy)
        min_period: Période minimale à considérer (points)
        max_period: Période maximale à considérer (points)
        acf_threshold: Seuil d'autocorrélation (0-1)
        fft_threshold: Seuil relatif pour les pics FFT
        max_detections: Nombre maximum de périodes à détecter
        
    Returns:
        periods_info: Liste de tuples (période, score, méthode) des périodes détectées
    """
    n = len(data)
    if max_period is None:
        max_period = n // 3  # Un tiers de la taille du signal
    
    print(f"Recherche de périodes entre {min_period} et {max_period} points (1/3 de la taille du signal)")
    
    max_lag = min(n // 2, max_period)
    periods_info = []
    # Pré-traitement pour réduire le bruit
    # Utilisons un filtre de Savitzky-Golay pour lisser légèrement les données
    try:
        data_smoothed = savgol_filter(data, 15, 3)  # Fenêtre de 15, polynôme d'ordre 3
    except:
        data_smoothed = data  # En cas d'échec, utiliser les données brutes
    
    # 1. Détection par autocorrélation
    # Normalisation des données pour l'autocorrélation
    data_norm = (data_smoothed - np.mean(data_smoothed)) / (np.std(data_smoothed) if np.std(data_smoothed) > 0 else 1)
    
    # Calcul de l'autocorrélation
    autocorr = np.correlate(data_norm, data_norm, mode='full')
    autocorr = autocorr[n-1:n-1+max_lag] / n  # Normalisation et sélection des lags positifs
    
    # Ignorer le premier pic (lag=0, autocorr=1)
    autocorr[0] = 0
    
    # Détection des pics d'autocorrélation avec une distance minimale
    peak_indices, peak_props = signal.find_peaks(
        autocorr, 
        height=acf_threshold,
        distance=min_period/2,  # S'assurer d'une séparation minimale entre pics
        prominence=0.1  # Requiert une certaine proéminence pour être considéré comme un pic
    )
    
    if len(peak_indices) > 0:
        # Filtrer les périodes selon les limites
        valid_peaks = [(idx, autocorr[idx]) for idx in peak_indices 
                      if min_period <= idx <= max_period]
        
        # Trier par importance
        valid_peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Ajouter à la liste des périodes avec méthode=ACF
        for period, score in valid_peaks[:max_detections//2]:
            periods_info.append((int(period), float(score), 'ACF'))
    
    # 2. Détection par analyse spectrale (FFT)
    # Détrend (enlève la tendance) pour une meilleure détection des cycles
    data_detrended = signal.detrend(data_smoothed)
    
    # Appliquer une fenêtre pour réduire les fuites spectrales
    window = signal.windows.hann(len(data_detrended))
    data_windowed = data_detrended * window
    
    # Calcul de la FFT
    yf = fft(data_windowed)
    xf = fftfreq(n, 1)[:n//2]  # Fréquences positives uniquement
    
    # Magnitude du spectre
    magnitude = 2.0/n * np.abs(yf[:n//2])
    
    # Ignorer les très basses fréquences (tendances)
    low_freq_cutoff = 1 / max_period
    magnitude_filtered = magnitude.copy()
    magnitude_filtered[xf < low_freq_cutoff] = 0
    
    # Détection des pics dans le spectre avec seuil adaptatif
    peak_indices_fft, peak_props_fft = signal.find_peaks(
        magnitude_filtered,
        height=np.max(magnitude_filtered) * fft_threshold,
        distance=5  # Éviter les pics trop proches en fréquence
    )
    
    fft_periods = []
    if len(peak_indices_fft) > 0:
        # Conversion des fréquences en périodes
        for i in peak_indices_fft:
            if xf[i] > 0:  # Éviter la division par zéro
                period = int(round(1 / abs(xf[i])))
                if min_period <= period <= max_period:
                    # Normaliser le score entre 0 et 1
                    score = magnitude[i] / np.max(magnitude)
                    fft_periods.append((period, score))
        
        # Trier par importance
        fft_periods.sort(key=lambda x: x[1], reverse=True)
        
        # Ajouter les périodes FFT les plus significatives
        for period, score in fft_periods[:max_detections//2]:
            periods_info.append((period, score, 'FFT'))
    
    # 3. Validation croisée: favoriser les périodes détectées par les deux méthodes
    validated_periods = []
    acf_periods = [p for p, _, m in periods_info if m == 'ACF']
    fft_periods = [p for p, _, m in periods_info if m == 'FFT']
    
    # Chercher des correspondances approximatives (±10%)
    for p_acf in acf_periods:
        for p_fft in fft_periods:
            # Si les périodes sont proches (différence < 10% de la plus petite)
            min_p = min(p_acf, p_fft)
            if abs(p_acf - p_fft) / min_p < 0.1:
                # Moyenne pondérée des deux périodes
                p_validated = int((p_acf + p_fft) / 2)
                
                # Chercher les scores correspondants
                acf_score = next((s for p, s, m in periods_info if p == p_acf and m == 'ACF'), 0)
                fft_score = next((s for p, s, m in periods_info if p == p_fft and m == 'FFT'), 0)
                
                # Score combiné avec bonus pour validation croisée
                combined_score = (acf_score + fft_score) * 1.2
                
                validated_periods.append((p_validated, combined_score, 'VALIDATED'))
    
    # Ajouter les périodes validées
    for p, s, m in validated_periods:
        # Éviter les doublons
        if not any(abs(p - existing_p) / min(p, existing_p) < 0.1 
                  for existing_p, _, existing_m in periods_info if existing_m == 'VALIDATED'):
            periods_info.append((p, s, m))
    
    # Trier l'ensemble des périodes par score
    periods_info.sort(key=lambda x: x[1], reverse=True)
    
    # Filtrer pour éliminer les périodes trop proches
    filtered_periods = []
    for period, score, method in periods_info:
        # Ne garder une période que si elle est suffisamment différente des périodes déjà retenues
        if not any(abs(period - p) / min(period, p) < 0.2 for p, _, _ in filtered_periods):
            filtered_periods.append((period, score, method))
    
    # Limiter au nombre maximum de détections souhaité
    filtered_periods = filtered_periods[:max_detections]
    
    # Trier par taille de période
    filtered_periods.sort(key=lambda x: x[0])
    
    # Si aucune période détectée, fournir des valeurs par défaut
    if not filtered_periods:
        filtered_periods = [(100, 1.0, 'DEFAULT'), (200, 0.8, 'DEFAULT'), (400, 0.6, 'DEFAULT')]
    filtered_periods.append([20,1.0,'Default'])
    filtered_periods.append([1,1.0,'Default'])
    return filtered_periods
