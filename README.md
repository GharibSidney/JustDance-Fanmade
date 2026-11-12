![Démo de JustDance-Fanmade avec la webcam et la chorégraphie côte à côte](images/dance_demo.png)

# JustDance-Fanmade

JustDance-Fanmade est un jeu de danse non officiel écrit en Python. Reproduisez les mouvements de la chorégraphie devant votre webcam. Le jeu détecte votre pose avec YOLO, la compare aux poses de référence et affiche une évaluation pendant la chanson.

## Installation

Il vous faut Python et une webcam. Depuis la racine du projet, installez les dépendances :

```powershell
python -m pip install -r requirements.txt
```

[Téléchargez les fichiers du projet sur SharePoint](https://polymtlca-my.sharepoint.com/:f:/g/personal/sidney_gharib_etud_polymtl_ca/IgDY8vg9BqqORrIe_oCFqKwUAb6w_5k6j4swtrOnrSm066o?e=0qK07N), puis extrayez les archives à la racine du projet. Vérifiez que les dossiers `labels` et `musics` ne sont pas imbriqués et que le modèle `yolo26n-pose.onnx` se trouve à côté de `dance.py`.

Pour la chanson jouée par défaut, les chemins attendus sont les suivants :

```text
JustDance-Fanmade/
├── dance.py
├── yolo26n-pose.onnx
├── labels/
│   └── Rasputin/
│       └── 0.json (et les autres fichiers de poses)
└── musics/
    └── Rasputin/
        ├── audio/Rasputin.mp3
        └── video/Rasputin.mp4
```

## Jouer

Depuis la racine du projet, lancez :

```powershell
python dance.py
```

La chanson **Rasputin** démarre avec la vue de la webcam et la vidéo de référence côte à côte. Appuyez sur **Q** dans la fenêtre du jeu pour quitter. Le score final s’affiche dans le terminal.

Le dépôt contient aussi du code commenté que vous pouvez réactiver pour expérimenter.
