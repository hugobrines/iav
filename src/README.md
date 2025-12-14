# Projet de Détection d'Anomalies sur Séries Temporelles

## Vue d'ensemble

Ce projet met en œuvre un système avancé de détection d'anomalies dans des signaux temporels en utilisant un modèle d'apprentissage profond de type TCN (Temporal Convolutional Network). L'architecture est conçue pour identifier des motifs périodiques, s'entraîner sur des données saines augmentées massivement, et détecter des déviations en temps réel.

Le workflow complet, de la génération des données à la visualisation des résultats, est détaillé ci-dessous.

## Flux de Travail (Workflow)

Pour utiliser ce projet, suivez les étapes dans l'ordre suivant.

### 1. Génération des Données de Test

La première étape consiste à générer les signaux synthétiques qui serviront à l'entraînement et à la validation du modèle.

- **Script à exécuter :** `LSTM/generate_test_signal.py`
- **Commande (depuis le répertoire `src/`) :**
  ```bash
  python3 LSTM/generate_test_signal.py
  ```

**AVERTISSEMENT :** L'exécution de ce script supprimera et remplacera tous les fichiers de signaux existants (`test_*.csv`) dans le répertoire `src/data/`.

### 2. Entraînement et Détection

Une fois les données générées, le script principal orchestre le processus d'entraînement et de détection.

- **Script à exécuter :** `LSTM/main.py`
- **Commande (depuis le répertoire `src/`) :**
  ```bash
  python3 LSTM/main.py
  ```
- **Processus :**
  1.  À son lancement, le script analyse le signal pour détecter les périodes potentielles.
  2.  Une interface graphique s'ouvre alors, affichant le signal. L'utilisateur doit **sélectionner graphiquement la période fondamentale** à utiliser pour l'entraînement en ajustant la vue et en cliquant sur le bouton de validation.
  3.  Une fois la période validée, le processus d'entraînement commence.

**AVERTISSEMENT MATÉRIEL :** Le modèle d'entraînement est volumineux et exigeant en ressources. Il est optimisé pour tourner sur une carte graphique **NVIDIA T4 avec 16 Go de VRAM**. Son exécution sur un ordinateur personnel standard sans GPU adéquat n'est pas recommandée et risque d'échouer.

### 3. Visualisation des Résultats

Après la fin du processus de `main.py`, un fichier de résultats contenant les anomalies détectées est généré dans `src/result/`. Pour analyser ces résultats de manière interactive :

- **Script à exécuter :** `LSTM/visualiser_anomaly_csv.py`
- **Commande (depuis le répertoire `src/`) :**
  ```bash
  python3 LSTM/visualiser_anomaly_csv.py --original data/test_1.3.csv --analysis result/test_1.3_anomalies.csv
  ```
- **Description :** Ce script lance une interface interactive pour visualiser le signal original, les prédictions, et les anomalies détectées. Il est nécessaire de fournir les chemins vers le fichier de données original et le fichier d'analyse généré.

## Description des Fichiers (`src/LSTM/`)

- **`main.py`**
  - Point d'entrée principal du projet. Il orchestre la détection de période, l'entraînement du modèle et la détection d'anomalies en temps réel.

- **`generate_test_signal.py`**
  - Générateur de signaux temporels synthétiques. Crée des fichiers CSV complexes incluant des périodicités multiples, du bruit et différents types d'anomalies.

- **`data.py`**
  - Contient les fonctions utilitaires pour le chargement et le prétraitement des données depuis les fichiers CSV.

- **`DetectPeriodRobust.py`**
  - Implémente l'algorithme de détection automatique des périodes prédominantes dans un signal en utilisant une analyse de l'autocorrélation.

- **`PeriodSelectorWithZoom.py`**
  - Fournit l'interface graphique (GUI) permettant à l'utilisateur de visualiser le signal, de zoomer et de sélectionner manuellement la période à utiliser pour l'entraînement.

- **`enhanced_augmentation.py`**
  - Module clé pour l'augmentation de données. À partir de quelques périodes de signal sain, il génère un très grand nombre de variations (changement de phase, d'amplitude, ajout de bruit, etc.) pour créer un jeu de données d'entraînement robuste.

- **`RobustSequenceAnomalyDetector.py`**
  - Définit la classe principale du détecteur d'anomalies. Elle contient l'architecture du modèle TCN, les logiques d'entraînement (`fit`), de prédiction (`predict`), et de sauvegarde/chargement du modèle.

- **`AnomalyVisualizer.py`**
  - Gère la fenêtre de visualisation *en temps réel* qui s'affiche pendant la phase de détection de `main.py`, mettant à jour le graphique à mesure que de nouveaux points sont traités.

- **`visualiser_anomaly_csv.py`**
  - Script d'analyse *post-exécution*. Il charge un fichier de données et un fichier de résultats pour une exploration interactive détaillée, incluant un slider pour ajuster le seuil de détection.
