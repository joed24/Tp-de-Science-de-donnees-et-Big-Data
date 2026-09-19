# Notes de lecture — Diday (1971)

Référence : E. Diday, « Une nouvelle méthode en classification automatique et reconnaissance des formes : la méthode des nuées dynamiques », Revue de Statistique Appliquée, tome 19, n° 2, 1971, pp. 19–33.

Ces notes reprennent uniquement les éléments utilisés dans le rapport.

- **p. 20** : le problème est de générer une partition à partir d'un corps de données, en recherchant la ressemblance maximale à l'intérieur des parties et minimale entre parties différentes.
- **p. 20** : introduction des notations pour l'ensemble des objets, les parties et les fonctions de distance.
- **p. 21** : l'algorithme alterne calcul des distances, partitionnement en K classes et construction de nouvelles représentations/étalons.
- **p. 21** : présentation d'une fonction générale d'agrégation-écartement `R(x,i,L)` et d'exemples de choix du critère.
- **pp. 22–23** : définitions d'élément améliorant, optimum local et convergence ; propositions et théorème de convergence sous certaines propriétés du critère.
- **p. 23** : entrées du programme : données, nombre de classes K, nombre d'étalons par classe et éventuellement des étalons choisis a priori. Sans information a priori, les étalons peuvent être tirés automatiquement au hasard. Sorties : partition, noyaux/étalons, mesures d'homogénéité et valeur de la partition.
- **p. 24** : intérêt des distances et des répétitions avec des tirages de départ différents ; notion de classifiabilité et d'individus charnières.
- **p. 25** : discussion sur le nombre d'étalons et sur l'apparition possible de classes vides lorsque K est trop grand.
- **p. 25** : exemple artificiel montrant que plusieurs étalons peuvent former un « squelette » ou une sorte d'axe factoriel discret et conserver des formes allongées mieux qu'un simple centre de gravité.
- **pp. 31–32** : exemple d'application en neurologie et étude de la stabilité des groupes par plusieurs tirages de départ.

Le rapport distingue les éléments provenant directement de l'article de 1971 des choix d'implémentation réalisés pour satisfaire la consigne du TP (axes, distribution et structure composite).
