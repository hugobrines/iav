
import os
import copy
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from collections import deque
from statsmodels.tsa.seasonal import STL
# Définir le backend avant d'importer matplotlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.debugging.set_log_device_placement(False)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout,
    Conv1D, LayerNormalization, SpatialDropout1D,
    Add, Activation, GlobalAveragePooling1D, MultiHeadAttention
)
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from scipy.interpolate import CubicSpline
from enhanced_augmentation import AdvancedAugmenter

class ExecutionLogger:
    """
    Logger pour enregistrer toutes les informations d'exécution du détecteur.
    Permet à un LLM d'analyser la qualité des résultats.
    """
    def __init__(self, output_dir="out"):
        self.output_dir = output_dir
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Créer le dossier de sortie si nécessaire
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Structure des logs
        self.log_data = {
            "execution_id": self.execution_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "configuration": {},
            "training": [],
            "retraining_events": [],
            "anomalies": [],
            "statistics": {
                "total_points_processed": 0,
                "total_anomalies": 0,
                "training_duration_seconds": 0,
                "total_retraining_count": 0
            }
        }

    def log_configuration(self, config_dict):
        """Enregistre la configuration du détecteur"""
        self.log_data["configuration"] = {
            **config_dict,
            "timestamp": datetime.now().isoformat()
        }

    def log_training_start(self, data_info):
        """Enregistre le début d'un entraînement"""
        training_event = {
            "type": "initial_training",
            "start_time": datetime.now().isoformat(),
            "data_info": data_info,
            "end_time": None,
            "results": {}
        }
        self.log_data["training"].append(training_event)
        return len(self.log_data["training"]) - 1  # Retourne l'index

    def log_training_end(self, training_idx, results):
        """Enregistre la fin d'un entraînement"""
        if training_idx < len(self.log_data["training"]):
            self.log_data["training"][training_idx]["end_time"] = datetime.now().isoformat()
            self.log_data["training"][training_idx]["results"] = results

            # Calculer la durée
            start = datetime.fromisoformat(self.log_data["training"][training_idx]["start_time"])
            end = datetime.fromisoformat(self.log_data["training"][training_idx]["end_time"])
            duration = (end - start).total_seconds()
            self.log_data["statistics"]["training_duration_seconds"] = duration

    def log_retraining(self, retraining_info):
        """Enregistre un événement de réentraînement"""
        retraining_event = {
            "timestamp": datetime.now().isoformat(),
            "retraining_number": len(self.log_data["retraining_events"]) + 1,
            **retraining_info
        }
        self.log_data["retraining_events"].append(retraining_event)
        self.log_data["statistics"]["total_retraining_count"] += 1

    def log_anomaly(self, anomaly_info):
        """Enregistre une anomalie détectée"""
        anomaly_event = {
            "anomaly_number": len(self.log_data["anomalies"]) + 1,
            "timestamp": datetime.now().isoformat(),
            **anomaly_info
        }
        self.log_data["anomalies"].append(anomaly_event)
        self.log_data["statistics"]["total_anomalies"] += 1

    def log_point_processed(self):
        """Incrémente le compteur de points traités"""
        self.log_data["statistics"]["total_points_processed"] += 1

    def finalize(self):
        """Finalise les logs avant sauvegarde"""
        self.log_data["end_time"] = datetime.now().isoformat()

        # Calculer des statistiques finales
        if self.log_data["statistics"]["total_points_processed"] > 0:
            anomaly_rate = (self.log_data["statistics"]["total_anomalies"] /
                          self.log_data["statistics"]["total_points_processed"]) * 100
            self.log_data["statistics"]["anomaly_rate_percent"] = round(anomaly_rate, 2)

    def save(self, filename=None):
        """Sauvegarde les logs dans un fichier JSON"""
        self.finalize()

        if filename is None:
            filename = f"execution_log_{self.execution_id}.json"

        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)

        print(f"\n  Logs d'exécution sauvegardés dans: {output_path}")
        return output_path

    def get_summary(self):
        """Retourne un résumé textuel des logs"""
        summary = []
        summary.append(f"=== RÉSUMÉ D'EXÉCUTION ===")
        summary.append(f"ID: {self.execution_id}")
        summary.append(f"Début: {self.log_data['start_time']}")
        summary.append(f"Fin: {self.log_data['end_time']}")
        summary.append(f"\n--- STATISTIQUES ---")
        summary.append(f"Points traités: {self.log_data['statistics']['total_points_processed']}")
        summary.append(f"Anomalies détectées: {self.log_data['statistics']['total_anomalies']}")
        if "anomaly_rate_percent" in self.log_data["statistics"]:
            summary.append(f"Taux d'anomalies: {self.log_data['statistics']['anomaly_rate_percent']}%")
        summary.append(f"Réentraînements: {self.log_data['statistics']['total_retraining_count']}")
        summary.append(f"Durée entraînement: {self.log_data['statistics']['training_duration_seconds']:.2f}s")

        return "\n".join(summary)


class RollingScaler:
    """
    Normalisation adaptative avec fenêtre glissante.
    S'adapte automatiquement aux changements d'échelle du signal.
    """
    def __init__(self, window_size=1000, feature_range=(-1, 1)):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None

    def fit_transform(self, data):
        """Initialise le scaler avec un ensemble de données"""
        if isinstance(data, np.ndarray):
            data = data.flatten()

        # Remplir la fenêtre avec les données initiales
        for val in data:
            self.window.append(val)

        # Calculer min/max initiaux
        self.min_val = min(self.window)
        self.max_val = max(self.window)

        # Transformer toutes les données
        result = []
        for val in data:
            scaled = self._scale_value(val)
            result.append(scaled)

        return np.array(result)

    def transform(self, value):
        """Transforme une valeur en mettant à jour la fenêtre"""
        # Ajouter la nouvelle valeur à la fenêtre
        self.window.append(value)

        # Recalculer min/max sur la fenêtre courante
        self.min_val = min(self.window)
        self.max_val = max(self.window)

        # Scaler la valeur
        return self._scale_value(value)

    def _scale_value(self, value):
        """Scale une valeur vers la plage feature_range"""
        if self.min_val is None or self.max_val is None:
            return 0.0

        # Éviter la division par zéro
        if self.max_val - self.min_val < 1e-10:
            return self.feature_range[0]

        # Normalisation dans la plage [-1, 1] par défaut
        scaled = (value - self.min_val) / (self.max_val - self.min_val)
        scaled = scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled

    def inverse_transform(self, scaled_value):
        """Dénormalise une valeur ou un array"""
        if self.min_val is None or self.max_val is None:
            return scaled_value

        # Gérer à la fois scalaires et arrays
        is_scalar = np.isscalar(scaled_value) or (isinstance(scaled_value, (int, float)))

        if is_scalar:
            # Cas scalaire
            value = (scaled_value - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
            value = value * (self.max_val - self.min_val) + self.min_val
            return value
        else:
            # Cas array
            scaled_array = np.array(scaled_value)
            value = (scaled_array - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
            value = value * (self.max_val - self.min_val) + self.min_val
            return value


class DriftDetector:
    """
    Détecteur de drift basé sur ADWIN (ADaptive WINdowing).
    Détecte les changements dans la distribution des erreurs.
    """
    def __init__(self, delta=0.002, window_size=100):
        self.delta = delta  # Seuil de confiance
        self.window_size = window_size
        self.error_window = deque(maxlen=window_size)
        self.mean_history = deque(maxlen=50)
        self.std_history = deque(maxlen=50)

    def add_error(self, error):
        """Ajoute une nouvelle erreur et détecte le drift"""
        self.error_window.append(error)

        # Calculer les statistiques de la fenêtre
        if len(self.error_window) >= 30:
            current_mean = np.mean(list(self.error_window)[-30:])
            current_std = np.std(list(self.error_window)[-30:])

            self.mean_history.append(current_mean)
            self.std_history.append(current_std)

            # Détection de drift : changement significatif dans la moyenne
            if len(self.mean_history) >= 10:
                old_mean = np.mean(list(self.mean_history)[:10])
                new_mean = np.mean(list(self.mean_history)[-10:])
                old_std = np.mean(list(self.std_history)[:10])

                # Drift détecté si la nouvelle moyenne diffère significativement
                threshold = old_std * 2 + self.delta
                if abs(new_mean - old_mean) > threshold:
                    return True

        return False

    def reset(self):
        """Réinitialise le détecteur après un drift"""
        self.error_window.clear()
        self.mean_history.clear()
        self.std_history.clear()



class RobustSequenceAnomalyDetector:
    def __init__(self, period_length, sequence_buffer_size=10, latent_dim=32, model_path=None,
                 use_stl_decomposition=True, enable_incremental_learning=True, enable_drift_detection=False, logger=None):
        """
        Initialise le détecteur d'anomalies robuste basé sur la reconnaissance de séquences périodiques.

        Args:
            period_length: Longueur de la période à apprendre (en nombre de points)
            sequence_buffer_size: Nombre de périodes complètes à conserver en mémoire
            latent_dim: Dimension de l'espace latent du modèle
            model_path: Chemin vers un modèle pré-entraîné (si None, un nouveau modèle est créé)
            use_stl_decomposition: Active la décomposition STL (Trend + Seasonal + Residual)
            enable_incremental_learning: Active l'apprentissage incrémental en ligne
            logger: Instance de ExecutionLogger pour enregistrer les événements
        """
        self.period_length = period_length
        self.sequence_buffer_size = sequence_buffer_size
        self.latent_dim = latent_dim
        self.use_stl_decomposition = use_stl_decomposition
        self.enable_incremental_learning = enable_incremental_learning
        self.enable_drift_detection = enable_drift_detection
        self.logger = logger

        # Logger la configuration si disponible
        if self.logger:
            self.logger.log_configuration({
                "period_length": period_length,
                "sequence_buffer_size": sequence_buffer_size,
                "latent_dim": latent_dim,
                "use_stl_decomposition": use_stl_decomposition,
                "enable_incremental_learning": enable_incremental_learning,
                "enable_drift_detection": enable_drift_detection,
                "model_path": model_path
            })

        # Buffer circulaire pour stocker les séquences complètes (utilisé pour l'apprentissage)
        self.buffer_capacity = period_length * sequence_buffer_size
        self.buffer = []

        # Préparation pour l'exécution
        self.is_trained = False
        self.current_sequence = []
        self.anomaly_threshold = None
        self.validation_errors = []

        # Utiliser le RollingScaler adaptatif au lieu du MinMaxScaler fixe
        self.scaler = RollingScaler(window_size=min(2000, period_length * 10), feature_range=(-1, 1))

        # Détecteur de drift pour adaptation automatique
        self.drift_detector = DriftDetector(delta=0.002, window_size=100)

        # Seuil adaptatif
        self.adaptive_errors = deque(maxlen=1000)  # Garder les 1000 dernières erreurs
        self.base_threshold = None  # Seuil de base calculé à l'entraînement

        # Composantes STL (si activée)
        self.stl_trend = deque(maxlen=5000)
        self.stl_seasonal = deque(maxlen=5000)
        self.stl_residual = deque(maxlen=5000)
        self.stl_fitted = False
        self.trend_model = None  # Modèle LSTM pour la tendance
        self.seasonal_model = None  # Modèle LSTM pour la saisonnalité

        # Encodage de phase (valeur + sin + cos)
        self.feature_dim = 3
        self.global_index = 0  # Compteur de points pour position de phase
        self.buffer_start_index = 0  # Phase du premier élément du buffer principal
        self.incremental_start_index = 0  # Phase du premier élément du buffer incrémental

        # Apprentissage incrémental
        self.incremental_buffer = []  # Buffer pour l'apprentissage incrémental
        self.incremental_batch_size = 32
        self.points_since_last_training = 0
        self.retraining_interval = period_length * 3  # Réentraîner tous les 3× période
        self.drift_count = 0
        self.min_incremental_buffer = period_length * 3  # Minimum 3× période pour réentraîner
        self.incremental_learning_enabled = enable_incremental_learning  # Flag pour désactiver si dégradation
        
        # Historique des points pour affichage
        self.recent_values = []         # Valeurs originales (avec anomalies) pour affichage
        self.recent_clean_values = []   # Valeurs nettoyées (sans anomalies) pour apprentissage
        self.recent_errors = []
        self.recent_predictions = []
        self.recent_times = []
        self.recent_anomaly_flags = []  # Flags indiquant si un point est une anomalie
        
        # Stocker la dernière valeur valide (non-anomalie)
        self.last_valid_value = None           # Valeur brute
        self.last_valid_scaled_value = None    # Valeur normalisée
        self.consecutive_anomalies = 0         # Compteur d'anomalies consécutives
        self.time_step = None  # Pas temporel constant entre deux points
        self.time_step_calculated = False  # Indicateur si le pas a été calculé

        self.input_length = self.period_length - 1
        self.use_attention = True
        
        # Création ou chargement du modèle
        if model_path and os.path.exists(model_path):
            print(f"Chargement du modèle depuis {model_path}")
            try:
                # Ajout des fonctions de perte personnalisées lors du chargement
                custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
                self.model = load_model(model_path, custom_objects=custom_objects)
                
                # Charger également le scaler si disponible
                scaler_path = model_path.replace('.h5', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    print(f"Scaler chargé depuis {scaler_path}")
                else:
                    # Si le scaler n'existe pas, créez-en un par défaut et ajustez-le sur quelques données
                    print("Aucun scaler trouvé, initialisation du scaler...")
                    self.scaler = MinMaxScaler(feature_range=(-1, 1))
                    # Utiliser des valeurs par défaut pour le fit initial
                    self.scaler.fit(np.array([[0], [100]]))
                    
                self.is_trained = True
            except Exception as e:
                print(f"Erreur lors du chargement du modèle: {e}")
                print("Création d'un nouveau modèle...")
                self.model = self._build_model_tcn()
                self.is_trained = False
        else:
            self.model = self._build_model_tcn()
            self.global_index = 0

    def _phase_encoding(self, length, start_pos):
        """Calcule sin/cos de phase pour chaque pas de temps."""
        positions = (np.arange(length) + start_pos) % self.period_length
        phase = positions / float(self.period_length)
        phase_sin = np.sin(2 * np.pi * phase)
        phase_cos = np.cos(2 * np.pi * phase)
        return phase_sin, phase_cos

    def _add_phase_encoding(self, sequence, start_pos):
        """Empile valeur + sin/cos de phase (shape: len(seq) x 3)."""
        seq = np.array(sequence).reshape(-1)
        phase_sin, phase_cos = self._phase_encoding(len(seq), start_pos)
        return np.stack([seq, phase_sin, phase_cos], axis=-1)

    def _frequency_perturbation(self, sequence, noise_level=0.05):
        """Léger bruit dans le domaine fréquentiel (amplitude uniquement)."""
        fft_vals = np.fft.rfft(sequence)
        noise = 1 + np.random.normal(0, noise_level, fft_vals.shape)
        fft_perturbed = fft_vals * noise
        return np.fft.irfft(fft_perturbed, n=len(sequence))

    def _tcn_residual_block(self, x, filters, kernel_size, dilation_rate, block_name):
        """
        Bloc résiduel TCN de base :
        Conv1D (causale) -> activation -> dropout -> Conv1D -> LayerNorm -> ajout résiduel -> activation
        """
        # Branche principale
        conv1 = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            name=f"{block_name}_conv1"
        )(x)
        conv1 = Activation("relu", name=f"{block_name}_relu1")(conv1)
        conv1 = SpatialDropout1D(0.2, name=f"{block_name}_dropout")(conv1)

        conv2 = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            name=f"{block_name}_conv2"
        )(conv1)
        conv2 = LayerNormalization(name=f"{block_name}_ln")(conv2)

        if self.use_attention:
            attn = MultiHeadAttention(
                num_heads=4,
                key_dim=max(8, filters // 8),
                name=f"{block_name}_attn"
            )(conv2, conv2)
            attn = SpatialDropout1D(0.2, name=f"{block_name}_attn_dropout")(attn)
            conv2 = Add(name=f"{block_name}_attn_add")([conv2, attn])

        # Branche résiduelle (adapter la dimension si nécessaire)
        if x.shape[-1] != filters:
            shortcut = Conv1D(
                filters=filters,
                kernel_size=1,
                padding="same",
                name=f"{block_name}_shortcut"
            )(x)
        else:
            shortcut = x

        out = Add(name=f"{block_name}_add")([shortcut, conv2])
        out = Activation("relu", name=f"{block_name}_relu_out")(out)
        return out
    
    def _build_model_tcn(self):
        """
        Modèle TCN résiduel pour prédire le prochain point d'une séquence temporelle périodique.
        """
        # Entrée: séquence de taille self.input_length (= period_length-1)
        inputs = Input(shape=(self.input_length, self.feature_dim), name="input_layer")

        x = inputs

        # Stack de blocs TCN résiduels avec dilatation croissante
        dilations = [1, 2, 4, 8,16,32]  
        filters = 128
        kernel_size = 7

        for i, d in enumerate(dilations):
            x = self._tcn_residual_block(
                x,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=d,
                block_name=f"tcn_block_{i+1}"
            )

        # Agrégation temporelle : on résume la séquence en un vecteur latent
        x = GlobalAveragePooling1D(name="tcn_global_pool")(x)

        # Couche de code latent
        # SOLUTION pour éviter saturation des neurones :
        # - ReLU au lieu de tanh → Pas de saturation, meilleur gradient
        # - tanh peut saturer à -1/+1 et bloquer l'apprentissage
        # Source: "Vanishing Gradient Solutions" (Baeldung)
        # Source: "Activation Functions for Deep Learning" (GeeksforGeeks)
        encoded = Dense(self.latent_dim, activation="relu", name="latent_encoding")(x)

        # Prédiction du prochain point
        outputs = Dense(1, activation="linear", name="next_point_prediction")(encoded)

        model = Model(inputs, outputs)

        # SOLUTION pour éviter les prédictions constantes :
        # 1. Learning rate plus élevé au début (0.001 au lieu de 0.0005)
        #    → Permet d'apprendre plus vite et éviter les minima locaux
        # 2. Clipnorm pour éviter exploding gradients
        #    → Limite la norme du gradient à 1.0
        # Source: "Learning Rate Scheduling" (Neptune.ai)
        # Source: "Vanishing Gradient Solutions" (ResearchGate)
        optimizer = Adam(
            learning_rate=0.0005,      
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0              
        )
        model.compile(optimizer=optimizer, loss="mse")

        print(f"Modèle TCN prédictif créé pour période de {self.period_length} points")
        model.summary()
        return model
    
    def _build_model(self):
        """
        Construit un modèle LSTM optimisé pour prédire uniquement le prochain point
        d'une séquence temporelle périodique.
        """
        # Entrée: séquence de taille period_length avec encodage de phase
        inputs = Input(shape=(self.period_length, self.feature_dim), name="input_layer")
        
        # Encodeur LSTM avec taille réduite
        # LSTM standard (non bidirectionnel) pour réduire les paramètres
        x = LSTM(256, return_sequences=True, activation='tanh', recurrent_activation='sigmoid',
                recurrent_dropout=0.1, name="lstm_encoder_1")(inputs)
        x = Dropout(0.2)(x)
        
        # Couche intermédiaire LSTM - pas de return_sequences puisqu'on ne veut que la dernière sortie
        x = LSTM(256, return_sequences=False, activation='tanh', recurrent_activation='sigmoid',
                name="lstm_encoder_2")(x)
        
        # Couche de code latent (représentation compacte)
        encoded = Dense(self.latent_dim, activation='tanh', name="latent_encoding")(x)
        
        # Au lieu du décodeur qui reconstruit toute la séquence,
        # ajouter une couche Dense qui prédit directement le prochain point
        outputs = Dense(1, activation='linear', name="next_point_prediction")(encoded)
        
        # Création du modèle
        model = Model(inputs, outputs)
        
        # Optimiseur avec paramètres ajustés
        optimizer = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        
        # Compilation avec fonction de perte adaptée
        model.compile(optimizer=optimizer, loss='mse')
        
        print(f"Modèle LSTM prédictif créé pour période de {self.period_length} points")
        model.summary()
        
        return model

    def calculate_time_step(self, times):
        """
        Calcule le pas temporel à partir des 5 premiers points
        
        Args:
            times: Liste des horodatages
            
        Returns:
            Pas temporel en secondes
        """
        if len(times) < 5:
            return None
        
        # Calculer les différences entre points consécutifs
        time_diffs = []
        for i in range(1, 5):
            diff = (times[i] - times[i-1]).total_seconds()
            time_diffs.append(diff)
        
        # Vérifier que le pas est à peu près constant
        median_step = np.median(time_diffs)
        max_deviation = max([abs(d - median_step) for d in time_diffs])
        
        # Si la déviation est trop grande, avertir mais continuer
        if max_deviation > 0.1 * median_step:
            print(f"ATTENTION: Pas temporel non constant détecté. Variation: {max_deviation:.2f}s")
            print(f"Valeurs: {time_diffs}")
        
        print(f"Pas temporel calculé: {median_step:.2f} secondes")
        return median_step

    def preprocess_data(self, data):
        """Prétraite les données pour l'apprentissage avec fenêtrage glissant"""
        # Normalisation des données
        data_shaped = np.array(data).reshape(-1, 1)
        
        # Vérifier la plage des données avant normalisation
        print(f"Données brutes - Min: {np.min(data)}, Max: {np.max(data)}")
        
        scaled_data = self.scaler.fit_transform(data_shaped).flatten()
        
        # Vérifier la plage des fdonnées après normalisation
        print(f"Données normalisées - Min: {np.min(scaled_data)}, Max: {np.max(scaled_data)}")
        
        # Séparation en séquences avec fenêtrage glissant
        sequences = []
        step_size = max(1, self.period_length // 3000)  # Avance de 0.03% de la période
        for i in range(0, len(scaled_data) - self.period_length + 1, step_size):
            seq = scaled_data[i:i + self.period_length]
            sequences.append(self._add_phase_encoding(seq, start_pos=i % self.period_length))
        
        # Conversion en tableau numpy pour l'entraînement
        X = np.array(sequences).reshape(-1, self.period_length, self.feature_dim)
        return X, scaled_data

    def train(self, data, epochs=50, validation_split=0.15, batch_size=32, patience=10, 
                val_data=None, num_training_periods=None, replications_per_type=30):
            """
            Entraîne le modèle avec augmentation de données MASSIVE.
            
            Techniques d'augmentation (chacune × replications_per_type × shifts):
            1. Jittering (bruit gaussien)
            2. Scaling (×0.8, ×0.9, ×1.1, ×1.2) - 4 variantes
            3. Offset (±0.1, ±0.2, ±0.3) - 6 variantes
            4. Magnitude Warping (déformation d'amplitude)
            5. Time Warping (déformation temporelle)
            6. Frequency Perturbation (perturbation spectrale)
            7. Trend Injection (dérive linéaire)
            8. Combinaisons Scaling+Offset - 4 variantes
            9. Mixup (si >1 période disponible)
            
            Avec 1 période de 160 points et 30 réplications/type:
            → 19 types × 30 reps × 160 shifts = 91,200 samples !
            
            Args:
                data: Signal d'entraînement
                epochs: Nombre d'époques
                validation_split: Fraction pour validation
                batch_size: Taille des batchs
                patience: Patience early stopping
                val_data: Données de validation réelles
                num_training_periods: Nombre de périodes à utiliser (None = tout)
                replications_per_type: Réplications PAR TYPE d'augmentation (défaut: 30)
            """
            BASE_NOISE_LEVEL = 0.03
            
            print(f"{'='*60}")
            print(f"=== ENTRAÎNEMENT AVEC AUGMENTATION MASSIVE ===")
            print(f"{'='*60}")
            print(f"Période: {self.period_length} points")
            print(f"Réplications par type: {replications_per_type}")
            
            # Logger le début de l'entraînement
            training_idx = None
            if self.logger:
                training_idx = self.logger.log_training_start({
                    "data_length": len(data),
                    "period_length": self.period_length,
                    "base_noise_level": BASE_NOISE_LEVEL,
                    "replications_per_type": replications_per_type,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "num_training_periods": num_training_periods
                })
            
            # Normalisation des données
            _, scaled_data = self.preprocess_data(data)
            
            if len(scaled_data) < self.period_length:
                print("❌ Pas assez de données!")
                return False
            
            all_periods = []
            for i in range(0, len(scaled_data) - self.period_length + 1, self.period_length):
                segment = scaled_data[i:i+self.period_length]
                if len(segment) == self.period_length:
                    all_periods.append(segment)
            
            if not all_periods:
                print("❌ Pas de période complète!")
                return False
            
            total_periods_available = len(all_periods)
            
            if num_training_periods is None:
                base_periods = all_periods
                num_periods_used = total_periods_available
            else:
                num_periods_used = min(num_training_periods, total_periods_available)
                base_periods = all_periods[:num_periods_used]
            
            print(f"\n  Périodes:")
            print(f"   - Disponibles: {total_periods_available}")
            print(f"   - Utilisées: {num_periods_used}")
            print(f"   - Points: {num_periods_used * self.period_length}")
            
            
            augmenter = AdvancedAugmenter(
                base_noise_level=BASE_NOISE_LEVEL,
                replications_per_type=replications_per_type
            )
            
            num_types = 1 + 4 + 6 + 1 + 1 + 1 + 1 + 4  # 19 types de base
            if len(base_periods) > 1:
                num_types += 1  # Mixup
            
            expected_samples = num_types * replications_per_type * self.period_length * num_periods_used
            
            print(f"\n  Configuration augmentation:")
            print(f"   - Types d'augmentation: {num_types}")
            print(f"   - Réplications/type: {replications_per_type}")
            print(f"   - Shifts/signal: {self.period_length}")
            print(f"   - Samples attendus: {expected_samples:,}")
            print(f"   - Facteur d'augmentation: {expected_samples / (num_periods_used * self.period_length):.0f}×")
            
           
            print(f"\n  Construction du dataset augmenté...")
            
            X_augmented = []
            y_augmented = []
            
            for period_idx, base_period in enumerate(base_periods):
                print(f"   Période {period_idx + 1}/{num_periods_used}...", end=" ")
                
                other_periods = [p for i, p in enumerate(base_periods) if i != period_idx]
                augmented_signals = []
                
                # 1. JITTERING SEUL
                for _ in range(replications_per_type):
                    aug = augmenter._add_jittering(base_period.copy())
                    augmented_signals.append(aug)
                
                # 2. SCALING (×0.8, ×0.9, ×1.1, ×1.2)
                for scale in augmenter.scaling_factors:
                    for _ in range(replications_per_type):
                        scaled = augmenter._apply_scaling(base_period.copy(), scale)
                        aug = augmenter._add_jittering(scaled)
                        augmented_signals.append(aug)
                
                # 3. OFFSET (±0.1, ±0.2, ±0.3)
                for offset in augmenter.offset_values:
                    for _ in range(replications_per_type):
                        offset_sig = augmenter._apply_offset(base_period.copy(), offset)
                        aug = augmenter._add_jittering(offset_sig)
                        augmented_signals.append(aug)
                
                # 4. MAGNITUDE WARPING
                for _ in range(replications_per_type):
                    warped = augmenter._apply_magnitude_warping(base_period.copy())
                    aug = augmenter._add_jittering(warped)
                    augmented_signals.append(aug)
                
                # 5. TIME WARPING
                for _ in range(replications_per_type):
                    warped = augmenter._apply_time_warping(base_period.copy())
                    aug = augmenter._add_jittering(warped)
                    augmented_signals.append(aug)
                
                # 6. FREQUENCY PERTURBATION
                for _ in range(replications_per_type):
                    perturbed = augmenter._apply_frequency_perturbation(base_period.copy())
                    aug = augmenter._add_jittering(perturbed)
                    augmented_signals.append(aug)
                
                # 7. TREND INJECTION (dérive)
                for _ in range(replications_per_type):
                    with_trend = augmenter._apply_trend_injection(base_period.copy())
                    aug = augmenter._add_jittering(with_trend)
                    augmented_signals.append(aug)
                
                # 8. COMBINAISONS SCALING + OFFSET
                combo_configs = [(0.9, 0.1), (1.1, -0.1), (0.8, 0.2), (1.2, -0.2)]
                for scale, offset in combo_configs:
                    for _ in range(replications_per_type):
                        combo = augmenter._apply_scaling(base_period.copy(), scale)
                        combo = augmenter._apply_offset(combo, offset)
                        aug = augmenter._add_jittering(combo)
                        augmented_signals.append(aug)
                
                # 9. MIXUP (si plusieurs périodes)
                if other_periods:
                    for _ in range(replications_per_type):
                        other = other_periods[np.random.randint(len(other_periods))]
                        mixed = augmenter._apply_mixup(base_period.copy(), other)
                        aug = augmenter._add_jittering(mixed)
                        augmented_signals.append(aug)
                
                print(f"{len(augmented_signals)} signaux", end=" → ")
                
                # Créer tous les shifts pour chaque signal augmenté
                samples_count = 0
                for aug_signal in augmented_signals:
                    for shift in range(self.period_length):
                        shifted = np.roll(aug_signal, -shift)
                        input_seq = shifted[:-1]
                        target = shifted[-1]
                        
                        encoded = self._add_phase_encoding(input_seq, start_pos=shift)
                        X_augmented.append(encoded)
                        y_augmented.append(target)
                        samples_count += 1
                
                print(f"{samples_count:,} samples")
            
            # Convertir en arrays
            X_augmented = np.array(X_augmented).reshape(-1, self.input_length, self.feature_dim)
            y_augmented = np.array(y_augmented).reshape(-1, 1)
            
            total_samples = len(X_augmented)
            augmentation_factor = total_samples / (num_periods_used * self.period_length)
            
            print(f"\n  Dataset augmenté créé:")
            print(f"   - Échantillons: {total_samples:,}")
            print(f"   - Facteur: {augmentation_factor:.0f}×")
            print(f"   - Shape X: {X_augmented.shape}")
            print(f"   - Shape y: {y_augmented.shape}")
            print(f"   - Mémoire: {(X_augmented.nbytes + y_augmented.nbytes) / 1024 / 1024:.1f} MB")
            
            
            print(f"\n Entraînement...")
            print(f"   - Epochs: {epochs}")
            print(f"   - Batch size: {batch_size}")
            print(f"   - Patience: {patience}")
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=patience//3,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Validation réelle si fournie
            val_tensors = None
            if val_data is not None and len(val_data) > self.period_length:
                val_values = np.array(val_data).flatten()
                temp_scaler = copy.deepcopy(self.scaler)
                val_scaled = np.array([temp_scaler.transform(v) for v in val_values])
                X_val_real, y_val_real = [], []
                step_size_val = max(1, self.period_length // 20)
                for i in range(0, len(val_scaled) - self.period_length, step_size_val):
                    input_seq = val_scaled[i:i+self.input_length]
                    X_val_real.append(self._add_phase_encoding(input_seq, start_pos=i % self.period_length))
                    y_val_real.append(val_scaled[i+self.input_length])
                if X_val_real:
                    X_val_real = np.array(X_val_real).reshape(-1, self.input_length, self.feature_dim)
                    y_val_real = np.array(y_val_real).reshape(-1, 1)
                    val_tensors = (X_val_real, y_val_real)
                    print(f"   - Validation: {len(X_val_real)} samples réels")
            
            fit_kwargs = dict(
                x=X_augmented, y=y_augmented,
                epochs=epochs, batch_size=batch_size,
                callbacks=callbacks, verbose=1
            )
            if val_tensors:
                fit_kwargs["validation_data"] = val_tensors
            else:
                fit_kwargs["validation_split"] = validation_split
            
            history = self.model.fit(**fit_kwargs)
            
            # Résultats
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            epochs_trained = len(history.history['loss'])
            
            print(f"\n✓ Entraînement terminé: {epochs_trained} epochs")
            print(f"   - Train loss: {final_loss:.6f}")
            print(f"   - Val loss: {final_val_loss:.6f}")
            print(f"   - Ratio: {final_val_loss/final_loss:.1f}×")
            
            print("\n  Calcul du seuil d'anomalie...")
            
            X_val, y_val = [], []
            step_size = max(1, self.period_length // 20)
            for i in range(0, len(scaled_data) - self.period_length, step_size):
                input_seq = scaled_data[i:i+self.input_length]
                X_val.append(self._add_phase_encoding(input_seq, start_pos=i % self.period_length))
                y_val.append(scaled_data[i+self.input_length])
            
            if X_val:
                X_val = np.array(X_val).reshape(-1, self.input_length, self.feature_dim)
                y_val = np.array(y_val).reshape(-1, 1)
                y_pred = self.model.predict(X_val, verbose=0)
                val_errors = np.square(y_val - y_pred).flatten()
                
                mean_error = np.mean(val_errors)
                std_error = np.std(val_errors)
                self.anomaly_threshold = mean_error + 3 * std_error
                self.base_threshold = self.anomaly_threshold
                
                print(f"   - Mean error: {mean_error:.6f}")
                print(f"   - Std error: {std_error:.6f}")
                print(f"   - Seuil (μ+3σ): {self.base_threshold:.6f}")
            else:
                self.anomaly_threshold = 0.1
                self.base_threshold = 0.1
            
            # Finalisation
            self.buffer = list(scaled_data[-self.buffer_capacity:])
            self.global_index = len(self.buffer)
            self.is_trained = True
            self.save_model("model_periods.h5")
            
            # Logger
            if self.logger and training_idx is not None:
                self.logger.log_training_end(training_idx, {
                    "total_samples": int(total_samples),
                    "augmentation_factor": float(augmentation_factor),
                    "final_loss": float(final_loss),
                    "final_val_loss": float(final_val_loss),
                    "epochs_trained": int(epochs_trained),
                    "anomaly_threshold": float(self.base_threshold),
                    "periods_used": int(num_periods_used),
                    "num_augmentation_types": int(num_types),
                    "replications_per_type": int(replications_per_type)
                })
            
            print(f"\n{'='*60}")
            print(f"  RÉSUMÉ FINAL")
            print(f"{'='*60}")
            print(f"   Périodes: {num_periods_used}/{total_periods_available}")
            print(f"   Types d'augmentation: {num_types}")
            print(f"   Samples entraînés: {total_samples:,}")
            print(f"   Facteur d'augmentation: {augmentation_factor:.0f}×")
            print(f"   Train/Val loss ratio: {final_val_loss/final_loss:.2f}×")
            print(f"{'='*60}")
            
            return True

    def save_model(self, path):
        """Sauvegarde le modèle entraîné et le scaler"""
        try:
            self.model.save(path, save_format='h5')
            print(f"Modèle sauvegardé: {path}")
            
            # Sauvegarder le scaler
            scaler_path = path.replace('.h5', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler sauvegardé: {scaler_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")

    def get_current_sequence(self):
        """Récupère la séquence actuelle d'entrée pour le modèle prédictif TCN :taille = input_length = period_length - 1"""
        if len(self.buffer) < self.input_length:
            # Pas assez de données, retourner None
            return None
            
        # Prendre la dernière séquence complète
        start_pos = (self.buffer_start_index + len(self.buffer) - self.input_length) % self.period_length
        seq = np.array(self.buffer[-self.input_length:])
        encoded = self._add_phase_encoding(seq, start_pos=start_pos)
        return encoded.reshape(1, self.input_length, self.feature_dim)
        
    def process_point(self, time, value):
        """
        Traite un nouveau point et détecte s'il s'agit d'une anomalie.
        Adapté pour fonctionner avec un modèle qui prend period_length-1 points
        et prédit le point suivant.
        
        Args:
            time: Horodatage du point
            value: Valeur du point
                
        Returns:
            (predicted_value, error, is_anomaly, reconstructed_values)
        """
        # Vérifier si on a déjà calculé le pas temporel
        if not self.time_step_calculated and len(self.recent_times) >= 5:
            self.time_step = self.calculate_time_step(self.recent_times)
            self.time_step_calculated = True
        
        # Vérifier l'écart temporel si on a un pas temporel et au moins un point précédent
        use_prediction_instead = False
        if self.time_step is not None and len(self.recent_times) > 0:
            last_time = self.recent_times[-1]
            time_gap = (time - last_time).total_seconds()
            
            # Vérifier si l'écart est plus de 2 fois supérieur au pas normal
            if time_gap > 2 * self.time_step:
                use_prediction_instead = True
                print(f"Écart temporel important détecté: {time_gap:.2f}s > {2 * self.time_step:.2f}s")
                print(f"Utilisation de la prédiction au lieu de la valeur réelle")
        
        # Normalisation de la nouvelle valeur
        scaled_value = self.scaler.transform(value)
        
        # Phase d'initialisation - tous les points sont acceptés comme valides
        if len(self.recent_values) < 1:
            # Durant l'initialisation, accepter les points comme valides
            self.last_valid_value = value
            self.last_valid_scaled_value = scaled_value
            
            # Ajouter aux buffers et historiques
            self.buffer.append(scaled_value)
            self.global_index += 1
            self.recent_values.append(value)
            self.recent_clean_values.append(value)
            self.recent_times.append(time)
            self.recent_errors.append(0.0)  # Pas d'erreur pendant l'initialisation
            self.recent_predictions.append(value)  # La prédiction est la valeur elle-même
            self.recent_anomaly_flags.append(False)
            
            # Initialiser le buffer d'adaptation si nécessaire
            self.adaptation_errors = getattr(self, 'adaptation_errors', [])
            
            # Gestion des tailles de buffer et historiques
            self._manage_buffer_sizes()
            
            # Message quand la phase d'initialisation se termine
            if len(self.recent_values) == min(2000, self.period_length):
                print(f"Phase d'initialisation terminée après {min(2000, self.period_length)} points")
                print(f"Démarrage de la détection d'anomalies avec adaptation dynamique du seuil")
            
            return value, 0.0, False, None
        
        # Vérifier si le modèle est entraîné
        if not self.is_trained:
            # Si le modèle n'est pas entraîné, continuer à accepter les points comme valides
            self.last_valid_value = value
            self.last_valid_scaled_value = scaled_value
            
            # Ajouter au buffer d'apprentissage et aux historiques
            self.buffer.append(scaled_value)
            self.global_index += 1
            self.recent_values.append(value)
            self.recent_clean_values.append(value)
            self.recent_times.append(time)
            self.recent_anomaly_flags.append(False)
            
            # Gestion des tailles de buffer et historiques
            self._manage_buffer_sizes()
            
            return None, None, False, None
        
        # Obtenir la séquence actuelle pour prédiction (period_length-1 points)
        # Modification ici pour utiliser period_length-1 au lieu de period_length
        if len(self.buffer) < self.input_length:
            # Pas assez de données pour faire une prédiction
            # Continuer à accepter les points comme valides
            self.last_valid_value = value
            self.last_valid_scaled_value = scaled_value
            
            # Ajouter au buffer et historiques
            self.buffer.append(scaled_value)
            self.global_index += 1
            self.recent_values.append(value)
            self.recent_clean_values.append(value)
            self.recent_times.append(time)
            self.recent_anomaly_flags.append(False)
            
            # Gestion des tailles de buffer et historiques
            self._manage_buffer_sizes()
            
            return None, None, False, None
        
        # Prendre les (period_length-1) derniers points pour la prédiction avec encodage de phase
        input_values = np.array(self.buffer[-self.input_length:])
        start_pos = (self.buffer_start_index + len(self.buffer) - self.input_length) % self.period_length
        input_seq = self._add_phase_encoding(input_values, start_pos=start_pos).reshape(1, self.input_length, self.feature_dim)
        
        # Prédiction du prochain point
        start_time = datetime.now()
        predicted_point = self.model.predict(input_seq, verbose=0)
        predict_time = (datetime.now() - start_time).total_seconds()
        if predict_time > 0.1:  # Ne pas surcharger les logs
            print(f"Temps d'inférence: {predict_time:.3f} secondes")
        
        # Extraction de la valeur prédite
        predicted_value_scaled = predicted_point[0, 0]  # Format [batch, features]
        
        # Conversion de la prédiction en valeur réelle (dénormalisation)
        predicted_value = self.scaler.inverse_transform(predicted_value_scaled)
        
        # Si un écart temporel important est détecté, utiliser la prédiction
        if use_prediction_instead:
            # Remplacer la valeur par la prédiction
            value = predicted_value
            # Recalculer la valeur normalisée
            scaled_value = predicted_value_scaled
            print(f"Valeur remplacée par la prédiction: {value:.2f}")
        
        # Calcul de l'erreur (MSE) entre la nouvelle valeur et la valeur prédite
        error = np.square(scaled_value - predicted_value_scaled)
        
        # S'assurer qu'on a un seuil d'anomalie
        if self.anomaly_threshold is None:
            self.anomaly_threshold = 0.1  # Valeur par défaut
            print(f"Aucun seuil d'anomalie trouvé, utilisation de la valeur par défaut: {self.anomaly_threshold}")

        # Calculer le seuil adaptatif basé sur les erreurs récentes
        current_threshold = self.get_adaptive_threshold()

        # Détection d'anomalie avec seuil adaptatif
        is_anomaly = error > current_threshold
        
        if use_prediction_instead:
            is_anomaly = True
            print(f"Point marqué comme anomalie en raison d'un écart temporel important")

        # GESTION DES ANOMALIES AVEC APPRENTISSAGE PROGRESSIF
        if is_anomaly:
            value_to_add_to_buffer = scaled_value
            clean_value = value

            # Logger l'anomalie
            if self.logger:
                self.logger.log_anomaly({
                    "time": time.isoformat() if hasattr(time, 'isoformat') else str(time),
                    "value": float(value),
                    "predicted_value": float(predicted_value),
                    "error": float(error),
                    "threshold": float(current_threshold),
                    "threshold_exceeded_by": float(error - current_threshold),
                    "used_prediction_instead": bool(use_prediction_instead)
                })
        else:
            # Pas d'anomalie: mettre à jour la dernière valeur valide
            self.last_valid_value = value
            self.last_valid_scaled_value = scaled_value
            value_to_add_to_buffer = scaled_value
            clean_value = value
            self.consecutive_anomalies = 0  # Réinitialiser le compteur

            # Ajouter l'erreur aux historiques adaptatifs (seulement pour points normaux)
            self.adaptive_errors.append(error)

            # Détection de drift sur les points normaux (désactivable)
            if self.enable_drift_detection:
                drift_detected = self.drift_detector.add_error(error)
                if drift_detected:
                    self.drift_count += 1
                    print(f"\n*** DRIFT DÉTECTÉ (#{self.drift_count}) ***")
                    print(f"Changement significatif dans la distribution des erreurs")
                    print(f"Erreur actuelle: {error:.6f}, Seuil: {current_threshold:.6f}")

                    # Réentraînement incrémental si activé
                    if self.incremental_learning_enabled and len(self.incremental_buffer) >= self.min_incremental_buffer:
                        print(f"Déclenchement du réentraînement incrémental suite au drift...")
                        self._incremental_train()

            # Apprentissage incrémental sur les points normaux
            if self.incremental_learning_enabled:
                if len(self.incremental_buffer) == 0:
                    # Phase du premier point incrémental alignée sur l'index courant du buffer principal
                    self.incremental_start_index = (self.buffer_start_index + len(self.buffer)) % self.period_length
                self.incremental_buffer.append(scaled_value)
                self.points_since_last_training += 1

                # Réentraînement périodique
                if self.points_since_last_training >= self.retraining_interval:
                    if len(self.incremental_buffer) >= self.min_incremental_buffer:
                        self._incremental_train()
                    self.points_since_last_training = 0
        
        # Ajouter au buffer la valeur appropriée (originale ou corrigée)
        self.buffer.append(value_to_add_to_buffer)
        self.global_index += 1
        
        # Ajouter aux historiques pour visualisation et analyse
        self.recent_values.append(value)               # Valeur brute originale
        self.recent_clean_values.append(clean_value)   # Valeur nettoyée
        self.recent_times.append(time)
        self.recent_errors.append(error)
        self.recent_predictions.append(predicted_value)
        self.recent_anomaly_flags.append(is_anomaly)
        
        # Gestion des tailles de buffer et historiques
        self._manage_buffer_sizes()

        # Logger le point traité
        if self.logger:
            self.logger.log_point_processed()

        # Pour la visualisation, créer une reconstruction complète à partir du buffer
        # Inclure la prédiction actuelle comme dernier point
        if len(self.buffer) >= self.period_length:
            # Prendre les derniers points du buffer (period_length-1) + la prédiction
            buffer_points = self.buffer[-(self.period_length-1):]
            visualizer_points = np.array(buffer_points + [predicted_value_scaled]).reshape(-1, 1)
            reconstructed_values = self.scaler.inverse_transform(visualizer_points).flatten()
        else:
            # Si pas assez de données, utiliser des valeurs nulles
            reconstructed_values = np.zeros(self.period_length)
        
        return predicted_value, error, is_anomaly, reconstructed_values

    def _manage_buffer_sizes(self):
        """Gestion des tailles des buffers et historiques"""
        # Buffer d'apprentissage
        if len(self.buffer) > self.buffer_capacity:
            self.buffer.pop(0)
            self.buffer_start_index = (self.buffer_start_index + 1) % self.period_length
        
        # Historiques pour affichage
        max_history = 100000
        for hist_list in [self.recent_values, self.recent_clean_values, self.recent_times, 
                         self.recent_anomaly_flags]:
            if len(hist_list) > max_history:
                hist_list.pop(0)
        
        for hist_list in [self.recent_errors, self.recent_predictions]:
            if hist_list and len(hist_list) > max_history:
                hist_list.pop(0)
    
    def get_visual_data(self):
        """Retourne les données pour la visualisation"""
        return {
            'times': self.recent_times,
            'values': self.recent_values,          # Valeurs brutes avec anomalies
            'clean_values': self.recent_clean_values,  # Valeurs nettoyées sans anomalies
            'predictions': self.recent_predictions,
            'errors': self.recent_errors,
            'threshold': self.anomaly_threshold,
            'is_trained': self.is_trained,
            'anomaly_flags': self.recent_anomaly_flags
        }
 
    def save_anomalies_to_csv(self, original_filename):
        """
        Enregistre les anomalies détectées et les prédictions dans un fichier CSV.
        
        Args:
            original_filename: Nom du fichier CSV original d'entrée
        
        Returns:
            str: Chemin vers le fichier CSV de résultat
        """

        
        # Créer le dossier de résultats s'il n'existe pas
        if not os.path.exists('result'):
            os.makedirs('result')
        
        # Extraire le nom du fichier sans extension ni chemin
        base_filename = os.path.basename(original_filename)
        filename_without_ext = os.path.splitext(base_filename)[0]
        
        # Créer le nom du fichier de sortie
        output_filename = f"result/{filename_without_ext}_anomalies.csv"
        
        # Préparer les données
        if not self.recent_times:
            print("Aucune donnée à enregistrer")
            return None
        
        # Créer un DataFrame avec toutes les données
        data = {
            'Date': [],
            'Heure': [],
            'ValeurOriginale': self.recent_values,
            'Prediction': self.recent_predictions,
            'AnomalyScore': self.recent_errors,
            'SeuilAnomalie': [self.anomaly_threshold] * len(self.recent_values),
            'EstAnomalie': [1 if flag else 0 for flag in self.recent_anomaly_flags]
        }
        
        # Extraire Date et Heure des timestamps
        for t in self.recent_times:
            data['Date'].append(t.strftime('%d/%m/%Y'))
            data['Heure'].append(t.strftime('%H:%M:%S'))
        
        # Créer le DataFrame
        df = pd.DataFrame(data)
        
        # Enregistrer dans un fichier CSV
        df.to_csv(output_filename, sep=';', index=False)
        
        print(f"Anomalies et prédictions enregistrées dans {output_filename}")
        print(f"Nombre total de points: {len(self.recent_values)}")
        print(f"Nombre d'anomalies détectées: {sum(self.recent_anomaly_flags)}")
        
        return output_filename

    def analyze_anomaly(self, time, value, error):
        """Analyse approfondie d'une anomalie détectée"""
        severity = "Sévère" if error > 3 * self.anomaly_threshold else "Modérée"
        
        # Calculer la déviation par rapport à la prédiction
        current_seq = self.get_current_sequence()
        if current_seq is None:
            return {'severity': severity, 'message': "Données insuffisantes pour l'analyse"}
            
        # Obtenir la prédiction pour le prochain point
        predicted_point = self.model.predict(current_seq, verbose=0)
        
        # Dénormaliser la prédiction (scalar)
        expected_value = self.scaler.inverse_transform(predicted_point[0, 0])
        
        # Dénormaliser la séquence actuelle pour l'analyse
        actual_seq = self.scaler.inverse_transform(current_seq[0]).flatten()
        
        # Calculer les statistiques de la séquence actuelle
        mean_seq = np.mean(actual_seq)
        std_seq = np.std(actual_seq)
        
        # Vérifier si l'anomalie est un pic ou une valeur manquante
        deviation = (value - expected_value) / std_seq if std_seq > 0 else float('inf')
        
        # Type d'anomalie
        if abs(deviation) > 3:
            anomaly_type = "Pic extrême"
        elif abs(deviation) > 1.5:
            anomaly_type = "Déviation significative"
        else:
            anomaly_type = "Déviation mineure"
            
        return {
            'severity': severity,
            'type': anomaly_type,
            'deviation': deviation,
            'expected_value': expected_value,
            'message': f"Anomalie {severity} détectée: {anomaly_type}. Valeur attendue: {expected_value:.2f}"
        }

    def get_adaptive_threshold(self):
        """
        Calcule le seuil adaptatif basé sur les erreurs récentes.

        Returns:
            float: Seuil adaptatif ou base_threshold si pas assez de données
        """
        if len(self.adaptive_errors) < 30:
            # Pas assez de données, utiliser le seuil de base
            return self.base_threshold if self.base_threshold is not None else self.anomaly_threshold

        # Calculer le seuil adaptatif à partir des erreurs récentes
        errors_array = np.array(list(self.adaptive_errors))
        mean_error = np.mean(errors_array)
        std_error = np.std(errors_array)

        # Seuil adaptatif = moyenne + 3 * écart-type
        adaptive_threshold = mean_error + 6 * std_error

        return adaptive_threshold

    def _incremental_train(self):
        """
        Réentraîne le modèle de manière incrémentale sur le buffer de données récentes.

        CORRECTIONS CRITIQUES appliquées :
        1. Learning Rate très bas (0.00001) pour éviter corruption des poids
        2. Buffer minimum 3× période pour avoir assez de séquences
        3. Validation avant/après pour vérifier amélioration
        4. Restauration des poids si dégradation
        """
        # Vérification du buffer minimum (3× période au lieu de 1×)
        if len(self.incremental_buffer) < self.min_incremental_buffer:
            return

        print(f"\n[Apprentissage incrémental] Réentraînement sur {len(self.incremental_buffer)} points...")

        # Logger le début du réentraînement
        retraining_start_time = datetime.now()

        # Préparer les données d'entraînement
        data = np.array(self.incremental_buffer)

        # Créer des séquences d'entraînement
        X_train = []
        y_train = []

        buffer_start_pos = self.incremental_start_index % self.period_length
        for i in range(len(data) - self.input_length):
            window = data[i:i+self.input_length]
            start_pos = (buffer_start_pos + i) % self.period_length
            X_train.append(self._add_phase_encoding(window, start_pos=start_pos))
            y_train.append(data[i+self.input_length])

        if len(X_train) < 10:  # Au moins 10 séquences
            print(f"[Apprentissage incrémental] Pas assez de séquences ({len(X_train)} < 10)")
            return

        X_train = np.array(X_train).reshape(-1, self.input_length, self.feature_dim)
        y_train = np.array(y_train).reshape(-1, 1)

        # CORRECTION CRITIQUE #1 : Sauvegarder les poids AVANT réentraînement
        weights_before = self.model.get_weights()

        # CORRECTION CRITIQUE #2 : Calculer la loss AVANT réentraînement
        loss_before = self.model.evaluate(X_train, y_train, verbose=0)

        # CORRECTION CRITIQUE #3 : Recompiler avec LR TRÈS BAS (0.00001)
        from tensorflow.keras.optimizers import Adam
        incremental_optimizer = Adam(
            learning_rate=0.00001,  # 100× plus bas que le LR initial !
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0
        )
        self.model.compile(optimizer=incremental_optimizer, loss="mse")

        # Réentraînement avec peu d'époques
        self.model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=min(32, len(X_train)),
            verbose=0
        )

        # CORRECTION CRITIQUE #4 : Calculer la loss APRÈS réentraînement
        loss_after = self.model.evaluate(X_train, y_train, verbose=0)

        # CORRECTION CRITIQUE #5 : Vérifier si le réentraînement a AMÉLIORÉ le modèle
        improved = loss_after <= loss_before * 1.1
        if not improved:  # Si dégradation > 10%
            print(f"[Apprentissage incrémental] ❌ DÉGRADATION détectée !")
            print(f"  Loss avant: {loss_before:.6f}, après: {loss_after:.6f}")
            print(f"  → Restauration des poids précédents")

            # Restaurer les poids
            self.model.set_weights(weights_before)

            # Désactiver l'apprentissage incrémental après 3 dégradations
            if not hasattr(self, 'incremental_degradation_count'):
                self.incremental_degradation_count = 0
            self.incremental_degradation_count += 1

            if self.incremental_degradation_count >= 3:
                print(f"[Apprentissage incrémental] ⚠️  DÉSACTIVATION après {self.incremental_degradation_count} dégradations")
                self.incremental_learning_enabled = False
        else:
            print(f"[Apprentissage incrémental] ✓ Amélioration: {loss_before:.6f} → {loss_after:.6f}")
            if hasattr(self, 'incremental_degradation_count'):
                self.incremental_degradation_count = 0  # Reset si succès

        # Logger le réentraînement
        if self.logger:
            retraining_duration = (datetime.now() - retraining_start_time).total_seconds()
            self.logger.log_retraining({
                "buffer_size": len(self.incremental_buffer),
                "sequences_count": len(X_train),
                "loss_before": float(loss_before),
                "loss_after": float(loss_after),
                "improved": bool(improved),
                "weights_restored": not improved,
                "duration_seconds": float(retraining_duration),
                "degradation_count": getattr(self, 'incremental_degradation_count', 0)
            })

        # CORRECTION CRITIQUE #6 : Recompiler avec l'optimizer ORIGINAL
        original_optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        self.model.compile(optimizer=original_optimizer, loss="mse")

        # Vider le buffer après l'entraînement (garder seulement les derniers points)
        if len(self.incremental_buffer) > self.period_length:
            removed = len(self.incremental_buffer) - self.period_length
            self.incremental_start_index = (self.incremental_start_index + removed) % self.period_length
            self.incremental_buffer = self.incremental_buffer[-self.period_length:]
