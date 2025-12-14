"""
Système d'augmentation de données avancé pour signaux périodiques.
Optimisé pour la détection de PICS (anomalies ponctuelles) dans des signaux
qui peuvent avoir des dérives naturelles.

Objectif: Le modèle doit apprendre que les dérives graduelles sont NORMALES,
mais qu'un point isolé avec une valeur inattendue est une ANOMALIE.

Auteur: Généré pour le projet TCN Anomaly Detection
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Optional


class AdvancedAugmenter:
    """
    Augmenteur de données avancé pour signaux périodiques.
    
    Techniques implémentées:
    1. Jittering (bruit gaussien)
    2. Scaling (mise à l'échelle globale)
    3. Offset (décalage vertical)
    4. Magnitude Warping (déformation d'amplitude locale)
    5. Time Warping (déformation temporelle)
    6. Frequency Perturbation (perturbation spectrale)
    7. Trend Injection (injection de tendance linéaire)
    8. Mixup (interpolation entre périodes)
    """
    
    def __init__(self, 
                 base_noise_level: float = 0.03,
                 replications_per_type: int = 30,
                 noise_variation_range: float = 0.4):
        """
        Args:
            base_noise_level: Écart-type du bruit gaussien de base (σ)
            replications_per_type: Nombre de réplications par type d'augmentation
            noise_variation_range: Variation du bruit (0.4 = ±20% du niveau de base)
        """
        self.base_noise_level = base_noise_level
        self.replications_per_type = replications_per_type
        self.noise_variation_range = noise_variation_range
        
        # Configuration des augmentations
        self.scaling_factors = [0.8, 0.9, 1.1, 1.2]
        self.offset_values = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
        
    def _add_jittering(self, signal: np.ndarray) -> np.ndarray:
        """Ajoute du bruit gaussien avec variation."""
        noise_factor = 1.0 - self.noise_variation_range/2 + \
                       self.noise_variation_range * np.random.random()
        noise_level = self.base_noise_level * noise_factor
        noise = np.random.normal(0, noise_level, len(signal))
        return signal + noise
    
    def _apply_scaling(self, signal: np.ndarray, factor: float) -> np.ndarray:
        """
        Mise à l'échelle globale du signal.
        Simule des variations d'amplitude du capteur.
        """
        return signal * factor
    
    def _apply_offset(self, signal: np.ndarray, offset: float) -> np.ndarray:
        """
        Décalage vertical du signal.
        Simule une dérive de baseline ou un changement de niveau.
        """
        return signal + offset
    
    def _apply_magnitude_warping(self, signal: np.ndarray, 
                                  num_knots: int = 4,
                                  sigma: float = 0.2) -> np.ndarray:
        """
        Déformation d'amplitude locale via courbe multiplicative.
        
        Crée une courbe lisse aléatoire qui multiplie le signal,
        simulant des variations d'amplitude non-uniformes.
        
        Args:
            num_knots: Nombre de points de contrôle pour la spline
            sigma: Écart-type des facteurs multiplicatifs (autour de 1.0)
        """
        # Créer des points de contrôle aléatoires
        knot_positions = np.linspace(0, len(signal) - 1, num_knots + 2)
        knot_values = np.random.normal(1.0, sigma, num_knots + 2)
        
        # Interpolation cubique pour créer une courbe lisse
        spline = CubicSpline(knot_positions, knot_values)
        warp_curve = spline(np.arange(len(signal)))
        
        return signal * warp_curve
    
    def _apply_time_warping(self, signal: np.ndarray,
                            num_knots: int = 4,
                            sigma: float = 0.2) -> np.ndarray:
        """
        Déformation temporelle du signal.
        
        Étire/compresse localement le signal dans le temps,
        simulant de légères variations de fréquence.
        
        Args:
            num_knots: Nombre de points de contrôle
            sigma: Écart-type de la déformation temporelle
        """
        n = len(signal)
        
        # Créer une fonction de warping temporel
        knot_positions = np.linspace(0, n - 1, num_knots + 2)
        # Les déplacements sont cumulatifs pour garantir la monotonie
        random_warps = np.random.normal(0, sigma, num_knots + 2)
        random_warps[0] = 0  # Fixer le début
        random_warps[-1] = 0  # Fixer la fin
        
        # Créer les nouvelles positions (doivent rester monotones)
        cumulative_warp = np.cumsum(random_warps)
        cumulative_warp = cumulative_warp - cumulative_warp[0]  # Commencer à 0
        
        # Normaliser pour que la fin reste à n-1
        warped_positions = knot_positions + cumulative_warp * (n / (num_knots + 1))
        warped_positions = np.clip(warped_positions, 0, n - 1)
        
        # S'assurer que c'est monotone croissant
        for i in range(1, len(warped_positions)):
            if warped_positions[i] <= warped_positions[i-1]:
                warped_positions[i] = warped_positions[i-1] + 0.1
        
        # Interpolation pour obtenir le signal déformé
        original_positions = np.arange(n)
        spline = CubicSpline(warped_positions, signal[knot_positions.astype(int)])
        
        # Rééchantillonner aux positions originales
        try:
            warped_signal = spline(original_positions)
        except:
            # En cas d'erreur, retourner le signal original avec bruit
            warped_signal = signal
            
        return warped_signal
    
    def _apply_frequency_perturbation(self, signal: np.ndarray,
                                       noise_level: float = 0.05) -> np.ndarray:
        """
        Perturbation dans le domaine fréquentiel.
        
        Modifie légèrement les amplitudes des composantes fréquentielles,
        simulant des variations spectrales naturelles.
        """
        fft_vals = np.fft.rfft(signal)
        
        # Bruit multiplicatif sur les amplitudes
        amplitude_noise = 1 + np.random.normal(0, noise_level, fft_vals.shape)
        
        # Légère perturbation de phase (optionnel)
        phase_noise = np.random.normal(0, noise_level * 0.5, fft_vals.shape)
        
        fft_perturbed = fft_vals * amplitude_noise * np.exp(1j * phase_noise)
        
        return np.fft.irfft(fft_perturbed, n=len(signal))
    
    def _apply_trend_injection(self, signal: np.ndarray,
                                max_slope: float = 0.3) -> np.ndarray:
        """
        Injection d'une tendance linéaire.
        
        Ajoute une pente aléatoire au signal,
        simulant une dérive lente du capteur.
        """
        n = len(signal)
        slope = np.random.uniform(-max_slope, max_slope)
        trend = np.linspace(0, slope, n)
        return signal + trend
    
    def _apply_mixup(self, signal1: np.ndarray, signal2: np.ndarray,
                     alpha: float = 0.3) -> np.ndarray:
        """
        Interpolation entre deux signaux (Mixup).
        
        Combine deux périodes pour créer une nouvelle variation.
        
        Args:
            alpha: Paramètre de la distribution Beta (plus petit = plus proche de signal1)
        """
        # Échantillonner λ depuis Beta(alpha, alpha)
        lam = np.random.beta(alpha, alpha)
        return lam * signal1 + (1 - lam) * signal2
    
    def augment_period(self, base_period: np.ndarray,
                       other_periods: Optional[List[np.ndarray]] = None,
                       include_mixup: bool = True) -> List[Tuple[np.ndarray, str]]:
        """
        Génère toutes les augmentations pour une période donnée.
        
        Args:
            base_period: Période de référence normalisée
            other_periods: Autres périodes disponibles pour Mixup
            include_mixup: Inclure l'augmentation Mixup
            
        Returns:
            Liste de tuples (signal_augmenté, type_augmentation)
        """
        augmented_samples = []

        for _ in range(self.replications_per_type):
            aug_signal = self._add_jittering(base_period.copy())
            augmented_samples.append((aug_signal, 'jittering'))

        for scale_factor in self.scaling_factors:
            for _ in range(self.replications_per_type):
                scaled = self._apply_scaling(base_period.copy(), scale_factor)
                aug_signal = self._add_jittering(scaled)
                augmented_samples.append((aug_signal, f'scaling_{scale_factor}'))

        for offset in self.offset_values:
            for _ in range(self.replications_per_type):
                offset_signal = self._apply_offset(base_period.copy(), offset)
                aug_signal = self._add_jittering(offset_signal)
                augmented_samples.append((aug_signal, f'offset_{offset}'))

        for _ in range(self.replications_per_type):
            warped = self._apply_magnitude_warping(base_period.copy())
            aug_signal = self._add_jittering(warped)
            augmented_samples.append((aug_signal, 'magnitude_warp'))

        for _ in range(self.replications_per_type):
            warped = self._apply_time_warping(base_period.copy())
            aug_signal = self._add_jittering(warped)
            augmented_samples.append((aug_signal, 'time_warp'))

        for _ in range(self.replications_per_type):
            perturbed = self._apply_frequency_perturbation(base_period.copy())
            aug_signal = self._add_jittering(perturbed)
            augmented_samples.append((aug_signal, 'freq_perturb'))

        for _ in range(self.replications_per_type):
            with_trend = self._apply_trend_injection(base_period.copy())
            aug_signal = self._add_jittering(with_trend)
            augmented_samples.append((aug_signal, 'trend'))

        combo_configs = [
            (0.9, 0.1),   # Légèrement réduit + décalé vers le haut
            (1.1, -0.1),  # Légèrement agrandi + décalé vers le bas
            (0.8, 0.2),   # Réduit + décalé fort vers le haut
            (1.2, -0.2),  # Agrandi + décalé fort vers le bas
        ]
        for scale, offset in combo_configs:
            for _ in range(self.replications_per_type):
                combo = self._apply_scaling(base_period.copy(), scale)
                combo = self._apply_offset(combo, offset)
                aug_signal = self._add_jittering(combo)
                augmented_samples.append((aug_signal, f'combo_s{scale}_o{offset}'))

        if include_mixup and other_periods and len(other_periods) > 0:
            for _ in range(self.replications_per_type):
                # Choisir une autre période aléatoirement
                other = other_periods[np.random.randint(len(other_periods))]
                mixed = self._apply_mixup(base_period.copy(), other)
                aug_signal = self._add_jittering(mixed)
                augmented_samples.append((aug_signal, 'mixup'))
        
        return augmented_samples
    
    def get_augmentation_summary(self, period_length: int, 
                                  num_base_periods: int = 1,
                                  include_mixup: bool = True) -> dict:
        """
        Calcule un résumé des augmentations qui seront générées.
        
        Returns:
            Dictionnaire avec les statistiques d'augmentation
        """
        # Compter les types d'augmentation
        num_scaling = len(self.scaling_factors)
        num_offset = len(self.offset_values)
        num_combos = 4  # Combinaisons scaling+offset
        
        types_count = {
            'jittering': 1,
            'scaling': num_scaling,
            'offset': num_offset,
            'magnitude_warp': 1,
            'time_warp': 1,
            'freq_perturb': 1,
            'trend': 1,
            'combos': num_combos,
        }
        
        if include_mixup and num_base_periods > 1:
            types_count['mixup'] = 1
        
        total_types = sum(types_count.values())
        
        # Calculer le nombre total de samples
        augmented_per_period = total_types * self.replications_per_type
        samples_per_period = augmented_per_period * period_length  # × shifts
        total_samples = samples_per_period * num_base_periods
        
        return {
            'types_count': types_count,
            'total_augmentation_types': total_types,
            'replications_per_type': self.replications_per_type,
            'augmented_signals_per_period': augmented_per_period,
            'samples_per_period': samples_per_period,
            'total_samples': total_samples,
            'augmentation_factor': samples_per_period / period_length
        }


def create_augmented_dataset(base_periods: List[np.ndarray],
                              period_length: int,
                              phase_encoding_fn,
                              replications_per_type: int = 30,
                              base_noise_level: float = 0.03,
                              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée un dataset d'entraînement complet avec toutes les augmentations.
    
    Args:
        base_periods: Liste des périodes de référence (normalisées)
        period_length: Longueur d'une période
        phase_encoding_fn: Fonction pour ajouter l'encodage de phase
        replications_per_type: Nombre de réplications par type d'augmentation
        base_noise_level: Niveau de bruit de base
        verbose: Afficher les informations de progression
        
    Returns:
        X: Array des séquences d'entrée (N, period_length-1, 3)
        y: Array des cibles (N, 1)
    """
    augmenter = AdvancedAugmenter(
        base_noise_level=base_noise_level,
        replications_per_type=replications_per_type
    )
    
    # Afficher le résumé
    if verbose:
        summary = augmenter.get_augmentation_summary(
            period_length, 
            len(base_periods),
            include_mixup=(len(base_periods) > 1)
        )
        print(f"\n  Configuration d'augmentation:")
        print(f"   - Types d'augmentation: {summary['total_augmentation_types']}")
        print(f"   - Réplications par type: {summary['replications_per_type']}")
        print(f"   - Signaux augmentés par période: {summary['augmented_signals_per_period']}")
        print(f"   - Samples par période (avec shifts): {summary['samples_per_period']:,}")
        print(f"   - Total samples attendu: {summary['total_samples']:,}")
        print(f"   - Facteur d'augmentation: {summary['augmentation_factor']:.0f}×")
        print(f"\n   Détail par type:")
        for type_name, count in summary['types_count'].items():
            samples = count * replications_per_type * period_length * len(base_periods)
            print(f"     • {type_name}: {count} variante(s) → {samples:,} samples")
    
    X_all = []
    y_all = []
    input_length = period_length - 1
    
    # Pour chaque période de base
    for period_idx, base_period in enumerate(base_periods):
        if verbose:
            print(f"\n  Traitement période {period_idx + 1}/{len(base_periods)}...")
        
        # Autres périodes pour Mixup
        other_periods = [p for i, p in enumerate(base_periods) if i != period_idx]
        
        # Générer toutes les augmentations
        augmented_list = augmenter.augment_period(
            base_period, 
            other_periods,
            include_mixup=(len(other_periods) > 0)
        )
        
        if verbose:
            print(f"   ✓ {len(augmented_list)} signaux augmentés générés")
        
        # Pour chaque signal augmenté, créer tous les shifts
        for aug_signal, aug_type in augmented_list:
            for shift in range(period_length):
                # Décalage circulaire
                shifted = np.roll(aug_signal, -shift)
                
                # Entrée: N-1 premiers points
                input_seq = shifted[:-1]
                # Cible: dernier point
                target = shifted[-1]
                
                # Ajouter l'encodage de phase
                encoded_input = phase_encoding_fn(input_seq, start_pos=shift)
                
                X_all.append(encoded_input)
                y_all.append(target)
    
    # Convertir en arrays
    X = np.array(X_all).reshape(-1, input_length, 3)
    y = np.array(y_all).reshape(-1, 1)
    
    if verbose:
        print(f"\n  Dataset créé:")
        print(f"   - Shape X: {X.shape}")
        print(f"   - Shape y: {y.shape}")
        print(f"   - Mémoire: {(X.nbytes + y.nbytes) / 1024 / 1024:.1f} MB")
    
    return X, y

if __name__ == "__main__":
    # Test avec une période synthétique
    np.random.seed(42)
    
    period_length = 160
    t = np.linspace(0, 2 * np.pi, period_length)
    
    test_period = np.sin(t) + 0.3 * np.sin(3 * t) + 0.1 * np.sin(5 * t)
    test_period = (test_period - test_period.min()) / (test_period.max() - test_period.min()) * 2 - 1
    
    # Fonction d'encodage de phase simple
    def simple_phase_encoding(seq, start_pos):
        positions = (np.arange(len(seq)) + start_pos) % period_length
        phase = positions / float(period_length)
        phase_sin = np.sin(2 * np.pi * phase)
        phase_cos = np.cos(2 * np.pi * phase)
        return np.stack([seq, phase_sin, phase_cos], axis=-1)
    
    # Test de l'augmenteur
    augmenter = AdvancedAugmenter(replications_per_type=30)
    
    # Résumé
    summary = augmenter.get_augmentation_summary(period_length, 1)
    print("=== Résumé de l'augmentation ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Générer le dataset
    print("\n=== Génération du dataset ===")
    X, y = create_augmented_dataset(
        base_periods=[test_period],
        period_length=period_length,
        phase_encoding_fn=simple_phase_encoding,
        replications_per_type=30,
        verbose=True
    )