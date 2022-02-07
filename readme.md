# Dependencies

- NumPy, Matplotlib , PIL
- sklearn (dataset and some functions)
- CV2

# instructions 
*Trois méthodes* sont disponibles dans ce projet, elles sont indépendantes l'une des autres. 
de préférence veillez décommenter les méthodes une par une pour avoir des résultats séparés.   

La premiere fonction **kComparaison** consiste à comparer la précision de la méthode vis-à-vis des différents K (de 3 à 100 avec un pas de 2). Nous sommes arrivé à la conclusion que la précision était la plus élevée pour un K égale à 3. 

La deuxième fonction **InternPrediction** consiste à essayer de prédire le chiffre inscrit sur une image provenant du dataset, la prediction est assez juste, vous pouvez inserer un parametre pour tester un nombre aléatoire différent d'images, par défaut, vous aurez 5.  

La troisième fonction **externPrediction** consiste à tester un image importer sur notre modèle, les résultats ne sont pas au rendez-vous, et nous avons eu du mal à avoir un résultat correct.  

Nous vous recommandons de tester chaque méthode séparement pour éviter un temps de compilement trop long. Seule la méthode **kComparaison** prend beaucoup de temps pour se terminer. 
