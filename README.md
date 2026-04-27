# ML_Project
Projet universitaire basé sur le projet Kaggle : [Birdclef-2026](https://www.kaggle.com/competitions/birdclef-2026/data)

**Auteurs**
- Yuyu CHEN
- Élise THOMAS

## Structure
```
├───augmentation/                   # données augmentées (audios en .wav)
│   │   train_audio/
├───data/                           # données originelles (audios en .ogg)
│   │   test_soundscapes/
|   |   train_audio/
|   |   train_soundscapes/
|   |   ...
├───eda_csv/                        # données réorganisées
│   │   eda_soundscapes_metada.csv
|   |   ...      
├───Images/                         # images .png des graphiques produits par les scripts
├───segments/                       # données segmentées (audios en .wav)
│   │   train_audio/
| create_augmentation_audio.py      # crée les fichiers .wav d'augmentation des segments d'audio
| create_segments_audio.py          # crée les fichiers .wav de segment des audios
| create_csv.ipynb                  # crée les csv stockés dans eda_csv/
| eda_data.ipynb                    # eda faite sur les données (fichiers audios)
| eda_metadata.ipynb                # eda faite sur les metadatas
| functions_eda.py                  # fonctions utilisées par les deux fichiers de eda
```

## Utilisation
1- Récupérer les datasets et les mettre dans data/
2- Lancer create_csv.ipynb
3- Lancer create_segments_audio.py
4- Lancer create_augmentation_audio.py