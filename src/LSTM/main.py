import traceback
from RobustSequenceAnomalyDetector import RobustSequenceAnomalyDetector, ExecutionLogger
from AnomalyVisualizer import AnomalyVisualizer
from DetectPeriodRobust import detect_periods_robust
from PeriodSelectorWithZoom import PeriodSelectorWithZoom
from data import load_data_from_csv,generate_stream_from_csv

def main():
    # Chemin vers le CSV
    # csv_filename = '../../data/BMOB25.csv'  # Modifier selon votre environnement
    csv_filename = 'data/test_1.csv'  # Fichier généré pour les tests

    model_path = 'model_periods.h5'  # Chemin pour sauvegarder/charger le modèle

    # Initialiser le logger d'exécution
    logger = ExecutionLogger(output_dir="out")
    print(f"  Logger d'exécution initialisé (ID: {logger.execution_id})")

    try:
        # Logger les informations initiales
        logger.log_data["input_file"] = csv_filename
        logger.log_data["model_path"] = model_path
        # Chargement des données
        print("Chargement des données...")
        data, times, _ = load_data_from_csv(csv_filename)
        
        # Détection automatique des périodes
        print("\nDétection des périodes sur tout le signal...")
        periods_info = detect_periods_robust(
            data,
            min_period=10,
            max_period=None,
            acf_threshold=0.3
        )
        
        print(f"Périodes détectées : {[p for p, _, _ in periods_info]}")
        
        # Interface de sélection de périodes
        print("\nOuverture de l'interface de sélection de périodes...")
        selector = PeriodSelectorWithZoom(data, times, periods_info)
        selected_periods, start_idx = selector.show()
        # main_period=60
        # En cas de période multiple, prendre la première pour l'instant
        main_period = selected_periods[0] if selected_periods else 100
        print(f"Période principale sélectionnée : {main_period}")
        print(f"Point de départ : {start_idx}")
        
        # Logger les périodes détectées
        logger.log_data["periods_detected"] = [int(p) for p, _, _ in periods_info]
        logger.log_data["selected_period"] = int(main_period)
        logger.log_data["start_index"] = int(start_idx)


        detector = RobustSequenceAnomalyDetector(
            period_length=main_period,
            sequence_buffer_size=5,  # Nombre de périodes à mémoriser
            latent_dim=32,
            model_path=None,  # Ne pas charger de modèle pré-entraîné
            use_stl_decomposition=True,       # ✓ Décomposition STL activée
            enable_incremental_learning=True,  # ✓ Apprentissage incrémental activé
            logger=logger  # Passer le logger au détecteur
        )
        
        # Toujours entraîner le modèle, peu importe s'il existe déjà
        train_data_full = data[start_idx:]
        val_data = None
        # Réserver un petit bloc réel pour validation/seuil (préférable à un split sur data augmentée)
        val_window = max(main_period * 3, 100)
        if len(train_data_full) > val_window + main_period:
            val_data = train_data_full[-val_window:]
            train_data = train_data_full[:-val_window]
        else:
            train_data = train_data_full

        print(f"\nEntraînement du détecteur sur {len(train_data)} points...")
        
        # S'assurer que le point de départ est aligné avec la période
        aligned_start = start_idx
        
        detector.train(
            train_data,
            epochs = 10 ,
            batch_size=64,
            patience=30,
            validation_split=0.15,
            val_data=val_data,
            num_training_periods=4 ,
            replications_per_type=30 
        )
        # Création du visualiseur
        visualizer = AnomalyVisualizer(detector)
        
        # Traitement du flux (à partir du point de départ)
        print("\nDémarrage de la détection d'anomalies en temps réel...")
        stream = generate_stream_from_csv(csv_filename, start_idx=0)

        anomaly_count = 0
        for t, value in stream:
            # Traitement du point
            predicted, error, is_anomaly, reconstruction = detector.process_point(t, value)
            
            # Si le détecteur n'est pas encore prêt
            if predicted is None:
                continue
                
            # Mise à jour de la visualisation (gérée par la classe)
            visualizer.update_plot()
                
            
            if is_anomaly:
                anomaly_count += 1
                anomaly_info = detector.analyze_anomaly(t, value, error)
                print(f"\n--- Anomalie n°{anomaly_count} détectée à {t} ---")
                print(f"Valeur: {value:.2f}, Prédiction: {predicted:.2f}, Erreur: {error:.6f}")
                print(f"Type: {anomaly_info['type']}, Sévérité: {anomaly_info['severity']}")
                print(f"Message: {anomaly_info['message']}")
                
                # Ajout à la visualisation
                is_severe = anomaly_info['severity'] == "Sévère"
                visualizer.add_anomaly(t, value, error, severe=is_severe)
            # Pause pour simulation temps réel (uniquement en démonstration)
        
        # Fin du traitement, maintenir la figure ouverte
        print("\nTraitement terminé.")
        print(f"Total d'anomalies détectées: {anomaly_count}")

        # Sauvegarder les résultats d'anomalies dans un CSV
        print("\nEnregistrement des résultats d'anomalies...")
        output_file = detector.save_anomalies_to_csv(csv_filename)
        print(f"Résultats enregistrés dans: {output_file}")
        logger.log_data["anomaly_csv_output"] = output_file

        # Sauvegarder les logs d'exécution
        print("\n" + logger.get_summary())
        log_file = logger.save()

        print(f"\n  Exécution terminée avec succès")
        print(f"   - Fichier de résultats: {output_file}")
        print(f"   - Fichier de logs: {log_file}")

        visualizer.show()
            
    except KeyboardInterrupt:
        print("\nTraitement interrompu par l'utilisateur.")
        logger.log_data["interrupted"] = True
        logger.save()
    except Exception as e:
        print(f"\nErreur: {e}")
        traceback.print_exc()
        logger.log_data["error"] = str(e)
        logger.log_data["traceback"] = traceback.format_exc()
        logger.save()
if __name__ == "__main__":
    main()
