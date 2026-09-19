# TP — Implémentation de l'algorithme des Nuées dynamiques (from scratch)

**Cours :** Science de données et Big Data  
**Étudiante :** Lumeya Kwivangana Exaucée  
**Promotion :** Master 1 Intelligence artificielle  
**Professeur :** Kafunda Jean-Pierre  
**Assisté par :** Grady Kamingu  
**Année académique :** 2025–2026

---

## 1. Objectif du projet

Ce dépôt contient une implémentation pédagogique de l'algorithme des **Nuées dynamiques**, sans utiliser une fonction de clustering provenant de `scikit-learn` ou d'une autre bibliothèque spécialisée.

L'application reprend le principe itératif général :

```text
Initialisation
     ↓
Affectation des individus aux classes
     ↓
Construction / mise à jour des représentations
     ↓
Calcul du critère
     ↓
Test de stabilisation
     ↓
Nouvelle itération si nécessaire
```

L'utilisateur peut choisir la représentation de chaque classe :

1. **ensemble de points représentatifs** ;
2. **axes factoriels** ;
3. **distribution probabiliste** ;
4. **structure représentative** ;
5. **point**, qui correspond au cas K-means.

> **Important :** les cinq modes forment une implémentation pédagogique cohérente avec la consigne du TP. Le premier article de Diday (1971) présente notamment des classes caractérisées par des *étalons* et une fonction générale d'agrégation-écartement. Les modes axes, distribution et structure sont ici des choix d'implémentation permettant d'explorer la généralité demandée par l'énoncé ; ils ne doivent pas être présentés comme une reproduction littérale de toutes les variantes théoriques de l'article original.

---

## 2. Référence scientifique principale

E. Diday, **« Une nouvelle méthode en classification automatique et reconnaissance des formes : la méthode des nuées dynamiques »**, *Revue de Statistique Appliquée*, tome 19, n° 2, 1971, pp. 19–33.

L'article original décrit notamment le problème de partitionnement, les notations, l'algorithme, la convergence, les entrées/sorties du programme et plusieurs applications. Il précise aussi qu'en l'absence d'information a priori, les étalons peuvent être tirés automatiquement au hasard.

Lien bibliographique :  
https://www.numdam.org/item/RSA_1971__19_2_19_0/

---

## 3. Contenu du projet

```text
nuees_project/
│
├── README.md
├── requirements.txt
├── setup_environment.py       # vérifie les dépendances avant toute installation
├── main.py                    # interface et lancement de l'application
├── run_windows.bat            # lancement rapide sous Windows
├── run_linux.sh               # lancement sous Linux/macOS
├── .gitignore
│
├── src/
│   ├── __init__.py
│   └── dynamic_clouds.py      # moteur complet from scratch
│
├── data/
│   ├── dataset.csv
│   └── dataset_demo_standardized.csv
│
├── tests/
│   └── test_algorithm.py
│
├── results/
│   ├── summary.csv
│   └── figures/
│       ├── *_partition.png
│       └── *_convergence.png
│
├── rapport/
│   ├── rapport.tex
│   ├── rapport.pdf
│   └── logo_unikin.png
│
└── references/
    └── README.md
```

Les fichiers `__pycache__`, l'environnement `.venv` et les résultats graphiques générés localement sont exclus du dépôt Git grâce au `.gitignore`.

---

# 4. Installation dans Visual Studio Code

## Étape 1 — Décompresser le ZIP

Décompresser le projet, puis ouvrir le dossier `nuees_project` dans **Visual Studio Code**.

Dans VS Code :

**File → Open Folder → nuees_project**

## Étape 2 — Ouvrir le terminal

Menu :

**Terminal → New Terminal**

Sous Windows PowerShell, vérifier :

```powershell
python --version
```

Une version récente de Python 3 est recommandée.

---

## 5. Dépendances : éviter les téléchargements inutiles

Le fichier `requirements.txt` contient la liste des dépendances nécessaires :

```text
numpy
pandas
matplotlib
```

**Ne lance pas obligatoirement `pip install -r requirements.txt`.**

Le projet fournit `setup_environment.py`, spécialement prévu pour éviter de télécharger des bibliothèques déjà installées.

Lancer :

```powershell
python setup_environment.py
```

Le script :

1. tente d'importer `numpy`, `pandas` et `matplotlib` ;
2. vérifie leur version ;
3. si tout est déjà correct, il ne télécharge rien ;
4. s'il manque une bibliothèque ou si une version minimale n'est pas respectée, il demande l'autorisation d'utiliser `pip` uniquement pour les dépendances concernées.

Exemple si tout est déjà installé :

```text
[OK] numpy: ...
[OK] pandas: ...
[OK] matplotlib: ...

Toutes les dépendances nécessaires sont déjà disponibles.
Aucun téléchargement n'a été effectué.
```

### À propos de l'environnement virtuel

Un environnement virtuel est recommandé mais pas obligatoire pour ce TP. Si `python -m venv .venv` fonctionne sur votre machine :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python setup_environment.py
```

Si Windows affiche une erreur lors de la création de `.venv`, il est possible de travailler directement avec l'installation Python existante et de lancer :

```powershell
python setup_environment.py
```

Puis le programme.

---

# 6. Lancer l'application

## Méthode interactive — recommandée pour la démonstration

```powershell
python main.py
```

Le programme affiche :

```text
============================================================
              CLASSIFICATION PAR NUÉES DYNAMIQUES
============================================================
1 - Ensemble de points représentatifs
2 - Axes factoriels
3 - Distribution probabiliste
4 - Structure représentative
5 - Point (K-Means)
6 - Exécuter les cinq représentations
============================================================
Votre choix [1-6] :
```

Puis il demande :

```text
Nombre de classes K [4] :
Nombre d'étalons/axes q [3] :
```

Pour reproduire l'expérience fournie dans le projet :

```text
Choix : 6
K : 4
q : 3
```

Le programme exécute les cinq représentations et produit les graphiques.

---

## 7. Lancer directement un mode

### K-means / prototype ponctuel

```powershell
python main.py --mode point --k 4 --seed 42
```

### Points représentatifs

```powershell
python main.py --mode points --k 4 --q 3 --seed 42
```

### Axes factoriels

```powershell
python main.py --mode axes --k 4 --q 3 --seed 42
```

En deux dimensions, le programme limite automatiquement le nombre d'axes utilisés afin de conserver un sous-espace non trivial.

### Distribution

```powershell
python main.py --mode distribution --k 4 --seed 42
```

### Structure représentative

```powershell
python main.py --mode structure --k 4 --q 3 --seed 42
```

### Tout exécuter

```powershell
python main.py --mode all --k 4 --q 3 --seed 42
```

---

# 8. Utiliser son propre fichier CSV

Le programme accepte un CSV contenant au moins deux colonnes numériques.

Exemple :

```text
age,revenu,score
22,500,0.72
25,620,0.81
31,900,0.65
...
```

Lancer :

```powershell
python main.py --mode points --k 3 --q 4 --data "data/mon_dataset.csv"
```

Les colonnes nommées `label`, `target` ou `class` sont ignorées pour l'apprentissage lorsqu'elles existent.

Les variables numériques sont standardisées :

\[
z_j=\frac{x_j-\bar{x}_j}{s_j}
\]

afin qu'une variable possédant une grande échelle ne domine pas artificiellement les distances.

---

# 9. Les cinq représentations

## 9.1 Point — K-means

Chaque classe est représentée par un seul centroïde :

\[
R_j=\mu_j
\]

avec :

\[
\mu_j=\frac{1}{|C_j|}\sum_{x_i\in C_j}x_i.
\]

La distance utilisée est la distance euclidienne au carré.

C'est le cas particulier dans lequel la représentation est réduite à un point.

## 9.2 Points représentatifs

Une classe est représentée par plusieurs individus appelés ici **étalons** :

\[
R_j=\{r_{j1},...,r_{jq}\}.
\]

La distance d'un individu à la classe est la distance au représentant le plus proche :

\[
d(x,R_j)=\min_{r\in R_j}\|x-r\|^2.
\]

Les représentants sont sélectionnés par une procédure gloutonne de couverture, implémentée directement avec NumPy.

## 9.3 Axes factoriels

La covariance de chaque classe est diagonalisée :

\[
S_jv=\lambda v.
\]

Les vecteurs propres associés aux plus grandes valeurs propres forment les axes retenus. La distance mesure alors la composante de l'individu qui reste hors du sous-espace représentatif.

## 9.4 Distribution probabiliste

La classe est décrite par une moyenne et une covariance :

\[
R_j=(\mu_j,\Sigma_j).
\]

La distance de Mahalanobis au carré est utilisée :

\[
d(x,R_j)=(x-\mu_j)^T\Sigma_j^{-1}(x-\mu_j).
\]

Une petite régularisation est ajoutée à la covariance avant inversion pour éviter les matrices singulières.

## 9.5 Structure représentative

La représentation combine trois informations :

- le centre ;
- la dispersion/covariance ;
- plusieurs étalons.

Les trois composantes sont normalisées puis combinées pour produire une distance composite.

---

# 10. Critère et convergence

Le programme utilise un critère opérationnel :

\[
J=\sum_{i=1}^{n}\min_j d(x_i,R_j).
\]

Après chaque reconstruction des prototypes, les individus sont réaffectés et le critère est recalculé.

L'algorithme s'arrête lorsque les affectations ne changent plus ou lorsque la variation relative du critère devient inférieure à la tolérance fixée.

Le nombre maximal d'itérations par défaut est 50.

**Attention :** les valeurs numériques du critère ne doivent pas être comparées directement entre les cinq modes comme s'il s'agissait de la même métrique : chaque représentation possède une fonction de distance différente.

---

# 11. Relation avec l'article de Diday (1971)

L'article fourni avec le TP formalise un ensemble d'objets `E`, une partition en classes et une fonction de distance/agrégation. Il décrit une construction itérative : calculer les distances, partitionner les individus, puis construire les nouveaux ensembles d'étalons.

L'article précise aussi que, sans information a priori, les étalons peuvent être tirés automatiquement au hasard. Le programme suit cette idée pour son initialisation.

L'article décrit enfin une fonction générale d'agrégation-écartement `R(x,i,L)` et une étude de convergence. Dans notre application pédagogique, nous retenons une forme simple de critère par distance au prototype, adaptée à chacun des cinq types demandés dans le TP.

Une figure de l'article montre également l'intérêt d'utiliser plusieurs étalons : pour des formes allongées, un seul centre de gravité peut « arrondir » la structure, alors que plusieurs étalons peuvent jouer le rôle d'un squelette ou d'un axe factoriel discret.

---

# 12. Résultats produits

Après :

```powershell
python main.py --mode all --k 4 --q 3 --seed 42
```

les fichiers suivants sont produits :

```text
results/summary.csv
results/figures/point_partition.png
results/figures/point_convergence.png
results/figures/points_partition.png
results/figures/points_convergence.png
results/figures/axes_partition.png
results/figures/axes_convergence.png
results/figures/distribution_partition.png
results/figures/distribution_convergence.png
results/figures/structure_partition.png
results/figures/structure_convergence.png
```

Le jeu de démonstration contient 400 observations en deux dimensions, réparties autour de quatre formes synthétiques.

Les figures permettent de voir les partitions finales ainsi que l'évolution du critère pour chaque représentation.

---

# 13. Tests

Lancer :

```powershell
python -m unittest discover -s tests -v
```

Les tests vérifient notamment que les cinq représentations peuvent être entraînées, que les prototypes sont construits et que `predict()` fonctionne.

---

# 14. Rapport LaTeX

Le rapport est disponible dans :

```text
rapport/rapport.tex
```

Une version PDF déjà compilée est fournie :

```text
rapport/rapport.pdf
```

Pour recompiler sous Windows avec une distribution LaTeX telle que MiKTeX :

```powershell
cd rapport
pdflatex rapport.tex
pdflatex rapport.tex
```

Le double passage permet de mettre correctement à jour la table des matières et les références internes.

---

# 15. Publier sur GitHub

Après avoir vérifié le projet :

```powershell
git init
git add .
git commit -m "TP Nuées dynamiques from scratch"
git branch -M main
git remote add origin https://github.com/TON_COMPTE/TON_REPO.git
git push -u origin main
```

Le lien GitHub à remettre avec le rapport sera alors celui de votre dépôt.

Ne mettez pas dans GitHub :

- `.venv/` ;
- `__pycache__/` ;
- les fichiers temporaires ;
- des fichiers personnels ou des mots de passe.

---

# 16. Démonstration conseillée devant l'enseignant

Pour une démonstration rapide :

```powershell
python setup_environment.py
python main.py --mode all --k 4 --q 3 --seed 42
```

Puis ouvrir :

```text
results/figures/
```

et montrer successivement :

1. le cas **Point / K-means** ;
2. les **points représentatifs** ;
3. les **axes factoriels** ;
4. la **distribution** ;
5. la **structure représentative**.

L'idée à expliquer est que le moteur reste le même : **affectation → représentation → réaffectation → stabilisation**, tandis que la fonction de représentation et de distance change.

---

## 17. Résumé

Le projet est volontairement organisé pour séparer :

- l'algorithme (`src/dynamic_clouds.py`) ;
- l'interface (`main.py`) ;
- les données (`data/`) ;
- les résultats (`results/`) ;
- les tests (`tests/`) ;
- la documentation (`README.md`) ;
- le rapport scientifique (`rapport/`).

L'implémentation du clustering est réalisée **from scratch** avec NumPy pour les calculs matriciels. Pandas sert au chargement CSV et Matplotlib à la visualisation.
