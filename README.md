# Projet ML Audio Classique — version structurée v7

Objectif : construire un pipeline ML classique pour reconnaissance d'espèces d'oiseaux à partir de `train_audio`, puis adaptation vers les `soundscapes` bruités et multi-label.

## Arborescence attendue

```text
projet/
├── data/
│   ├── train_audio/
│   ├── train_soundscapes/
│   ├── test_soundscapes/
│   ├── train.csv
│   ├── train_soundscapes_labels.csv
│   ├── sample_submission.csv
│   ├── taxonomy.csv
│   └── recording_location.txt
├── notebooks/
├── src/
└── outputs/
```

## Ordre d'exécution conseillé

1. `01_eda_audio_project_driven.ipynb`
2. `02_feature_extraction_cache_preprocessing.ipynb`
3. `03_baseline_and_classical_models.ipynb`
4. `04_save_and_compare_all_models.ipynb`
5. `05_error_analysis_top_models.ipynb`
6. `06_train_final_top3_models.ipynb`
7. `07_predict_train_soundscapes_domain_shift.ipynb`
8. `08_threshold_tuning_soundscapes.ipynb`
9. `09_predict_test_submission.ipynb`
10. `10_optional_dtw_refinement.ipynb`

## Idée importante

`train_audio` est propre et mono-espèce, alors que les soundscapes sont bruités, longs et multi-label. Le projet utilise donc :

- EDA orientée décision ;
- preprocessing audio robuste ;
- cache des features ;
- modèles ML classiques : SVM, SGD logistic, KNN, RandomForest, ExtraTrees, HistGradientBoosting, AdaBoost ;
- sauvegarde de tous les modèles baseline ;
- sélection top 3 ;
- segmentation + overlap pour soundscapes ;
- fusion des scores et tuning du seuil multi-label.
