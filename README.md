# 🚗 ROAD-TRACKING — Système de Détection & Suivi de Véhicules en Temps Réel

**Application Python de vision par ordinateur combinant YOLO v2.6 custom et OpenCV pour la détection, le tracking et l'analyse de trafic routier en conditions réelles**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v2.6_Custom-FF6B00?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-Detection-00B4D8?style=for-the-badge)](https://ultralytics.com/)
[![Real-Time](https://img.shields.io/badge/Real--Time-Processing-00C853?style=for-the-badge)]()

## 🌐 **[VOIR LE REPO](https://github.com/Ayman-cell/road-tracking)** 🌐

</div>

---

**ROAD-TRACKING** est une application de vision par ordinateur en Python permettant de détecter et suivre des véhicules en temps réel sur des vidéos de routes ou via flux caméra live, en utilisant un modèle YOLO personnalisé entraîné (`yolo26s.pt`).

Ce projet combine :

- 🎯 **Détection multi-classes** de véhicules (voitures, camions, motos, bus, piétons…)
- 🔄 **Tracking multi-objets** avec assignation d'IDs stables par frame
- 📹 **Support vidéo** fichier ou caméra en temps réel
- 🏋️ **Modèle YOLO custom** entraîné et optimisé : `yolo26s.pt`
- 📊 **Comptage de véhicules** avec ligne de franchissement
- 🎨 **Affichage bounding boxes** + labels + IDs en overlay
- ⚡ **Pipeline temps réel** frame-by-frame optimisé avec OpenCV
- 🔧 **Déploiement standalone** sans infrastructure cloud

---

# ✨ Fonctionnalités principales

## 1️⃣ Détection YOLO Haute Précision

- 🤖 **Modèle custom entraîné** : `yolo26s.pt` (YOLO v2.6 small — optimisé vitesse/précision)
- 🎯 **Détection multi-classes** : voitures, camions, motos, bus, piétons et plus
- 📦 **Bounding boxes** avec scores de confiance affichés en temps réel
- ⏱️ **Inférence rapide** : traitement frame-by-frame optimisé
- 🔧 **Seuil de confiance configurable** pour filtrer les fausses détections
- 📐 **NMS (Non-Maximum Suppression)** pour éviter les doublons de détection

**Modèle inclus :**
```
road-tracking/
└── yolo26s.pt    # Poids YOLO v2.6 small custom-trained
```

---

## 2️⃣ Tracking Multi-Objets Stable

- 🔢 **IDs uniques** assignés à chaque véhicule détecté et maintenus entre frames
- 🔄 **Re-identification** des objets entre frames consécutifs
- 🚦 **Résistance aux occlusions** partielles
- 📍 **Suivi de trajectoire** des véhicules à travers la scène
- 🎯 **Association par IoU** (Intersection over Union) entre détections successives

---

## 3️⃣ Analyse de Trafic en Temps Réel

- 📏 **Ligne de comptage virtuelle** configurable (ENTRY/EXIT line)
- 🔢 **Compteur de véhicules** en entrée et en sortie
- 🚗 **Classification par type** de véhicule
- 📊 **Overlay statistiques** en temps réel sur le flux vidéo
- ⚡ **Affichage FPS** et performance de traitement en direct

---

## 4️⃣ Pipeline Vidéo Flexible

- 📹 **Sources supportées** :
  - Fichier vidéo local (`.mp4`, `.avi`, `.mov`, etc.)
  - Flux caméra en direct (webcam, IP camera, RTSP)
- 🎨 **Rendu visuel** avec bounding boxes colorées par classe
- 💾 **Export vidéo** annotée avec les détections
- 🖥️ **Affichage fenêtre** temps réel avec `cv2.imshow`
- ⏸️ **Contrôles** : pause/reprise, quitter via `Esc`

---

# 🛠 Technologies utilisées

| Technologie | Utilisation | Version |
|-------------|-------------|---------|
| **Python** | Langage principal | 3.8+ |
| **Ultralytics YOLO** | Modèle de détection | Custom yolo26s |
| **OpenCV** | Traitement vidéo & affichage | 4.x+ |
| **NumPy** | Traitement arrays/matrices | Latest |
| **PyTorch** | Backend inférence YOLO | 1.x+ |

---

# 📊 Performances

| Métrique | Valeur |
|----------|--------|
| **Modèle** | YOLO v2.6 Small (yolo26s.pt) |
| **Classes détectées** | Véhicules + piétons |
| **Source vidéo** | Fichier ou caméra live |
| **Tracking** | Multi-objets avec IDs stables |
| **Mode** | Temps réel (Real Life) |

---

# 📂 Structure du projet

```
road-tracking/
│
├── 📄 real life.py       # Script principal — détection + tracking temps réel
├── 🤖 yolo26s.pt         # Modèle YOLO v2.6 small (poids entraînés custom)
└── 📄 README.md          # Documentation du projet
```

### Description des fichiers

**`real life.py`** — Le cœur de l'application. Ce script :
- Charge le modèle YOLO custom `yolo26s.pt`
- Ouvre un flux vidéo (fichier ou caméra)
- Effectue la détection frame par frame
- Applique le tracking multi-objets avec IDs persistants
- Dessine les bounding boxes, labels et IDs en overlay
- Affiche le flux annoté en temps réel via OpenCV
- Gère le comptage des véhicules via une ligne de franchissement

**`yolo26s.pt`** — Les poids du modèle YOLO v2.6 Small entraîné et optimisé pour la détection de véhicules en conditions routières réelles.

---

# 🚀 Installation & Démarrage

## Prérequis

- Python 3.8+
- pip
- GPU recommandé (NVIDIA CUDA) pour performances optimales
- Webcam ou fichier vidéo de test

---

## 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/Ayman-cell/road-tracking.git
cd road-tracking
```

---

## 2️⃣ Installer les dépendances

```bash
pip install ultralytics opencv-python numpy torch torchvision
```

Ou via un fichier requirements (si disponible) :

```bash
pip install -r requirements.txt
```

**Dépendances principales :**

```
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0
torch>=1.9.0
torchvision>=0.10.0
```

---

## 3️⃣ Lancer l'application

### Avec un fichier vidéo

Modifiez la ligne de source vidéo dans `real life.py` :

```python
# Vidéo fichier
cap = cv2.VideoCapture('votre_video.mp4')  # ← Chemin vers votre vidéo
```

Puis lancez :

```bash
python "real life.py"
```

### Avec une caméra en direct

```python
# Caméra (index 0 = webcam par défaut)
cap = cv2.VideoCapture(0)
```

Puis lancez :

```bash
python "real life.py"
```

### Avec une caméra IP / RTSP

```python
cap = cv2.VideoCapture('rtsp://192.168.1.100:554/stream')
```

---

## 4️⃣ Contrôles pendant l'exécution

| Touche | Action |
|--------|--------|
| `Esc` | Quitter l'application |
| `Space` | Pause / Reprendre |
| `s` | Sauvegarder la frame courante |

---

# ⚙️ Configuration

Paramètres configurables dans `real life.py` :

```python
# Seuil de confiance (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.5   # Augmenter pour moins de fausses détections

# Ligne de comptage (position Y en pixels)
COUNTING_LINE_Y = 300        # Adapter selon la résolution vidéo

# Tolérance de franchissement de ligne
LINE_TOLERANCE = 10          # En pixels

# Modèle YOLO à utiliser
MODEL_PATH = 'yolo26s.pt'    # Chemin vers les poids

# Affichage
SHOW_LABELS = True           # Afficher les labels de classe
SHOW_CONFIDENCE = True       # Afficher les scores de confiance
SHOW_TRACKING_ID = True      # Afficher les IDs de tracking
```

---

# 🔧 Dépannage

### Erreur : "No module named 'ultralytics'"
```bash
pip install ultralytics --upgrade
```

### Erreur : "No module named 'cv2'"
```bash
pip install opencv-python
```

### Fenêtre vidéo ne s'affiche pas (serveur headless)
```bash
# Utiliser un affichage virtuel
sudo apt-get install xvfb
Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99
python "real life.py"
```

### Performance lente (CPU uniquement)
```bash
# Installer PyTorch avec support CUDA (GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Le modèle `yolo26s.pt` non trouvé
```bash
# Vérifier que vous êtes dans le bon répertoire
ls -la yolo26s.pt

# Spécifier le chemin absolu si nécessaire
MODEL_PATH = '/chemin/absolu/vers/road-tracking/yolo26s.pt'
```

### Détections peu précises
- Vérifiez que `CONFIDENCE_THRESHOLD` est adapté à votre scène (essayez 0.3 - 0.6)
- Assurez-vous que la résolution d'entrée est correcte
- Vérifiez l'éclairage et la qualité de la source vidéo

---

# 🎯 Cas d'usage

- ✅ **Surveillance trafic routier** avec caméras fixes
- ✅ **Comptage de véhicules** sur route ou autoroute
- ✅ **Analyse de flux de circulation** en intersection
- ✅ **Détection d'intrusion** de véhicules en zones restreintes
- ✅ **Prototype de système** de gestion intelligente du trafic
- ✅ **Recherche en Computer Vision** appliquée au transport
- ✅ **Base de départ** pour projets de conduite autonome

---

# 🔮 Améliorations possibles

- 📡 **Intégration Deep SORT** pour tracking plus robuste avec Kalman Filter
- 🚀 **Estimation de vitesse** des véhicules en km/h
- 🎨 **Classification** par couleur et modèle du véhicule
- 📊 **Dashboard** temps réel des statistiques de trafic
- 💾 **Export données** en CSV/JSON pour analyse ultérieure
- 🌐 **Interface web** avec Flask pour streaming distant
- 📱 **API REST** pour intégration dans d'autres systèmes
- 🏋️ **Fine-tuning** du modèle sur données spécifiques à la scène

---

# 👨‍💻 Auteur

**Développé par : Ayman Amasrour — EMINES, UMP Benguerir**

| Rôle | Membre | Responsabilités |
|------|--------|-----------------|
| **AI/ML Engineer** | Ayman Amasrour | Architecture Computer Vision, Modèle YOLO custom, Pipeline détection/tracking, Optimisation temps réel |

---

# 📝 Licence

**Licence MIT** — Projet académique / Recherche

Développé dans le cadre d'un projet de recherche en vision par ordinateur à **EMINES, UMP Benguerir**.

---

<div align="center">

## **Détection intelligente de véhicules pour des routes plus sûres** 🛣️

**Road-Tracking** — Computer Vision + YOLO + OpenCV pour la mobilité intelligente

</div>
