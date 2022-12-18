# Importation des librairies
import pandas as pd
from surprise import SVD
import pickle
# Load the data from a CSV file

anime_df = pd.read_csv('./datas/anime.csv')
rating_df = pd.read_csv('./datas/rating.csv')

# Initialisation de l'algorithme SVD et récupération
algo = SVD()
with open('./saved_model.pkl', 'rb') as f:
    algo = pickle.load(f)


def get_top_10_recommendations(model, user_id):
    # Prédire les notes pour tous les films
    all_predictions = []
    for i in range(1, 34528):
        all_predictions.append(model.predict(user_id, i))

    # Trier les prédictions par note prédite décroissante
    all_predictions.sort(key=lambda x: x.est, reverse=True)

    # Garder les 10 meilleures prédictions
    top_10_predictions = all_predictions[:10]

    # Récupérer les IDs des films recommandés
    top_10_movie_ids = [pred[1] for pred in top_10_predictions]

    return top_10_movie_ids


def get_names(ids):
    names = []
    for id in ids:
        names.append(
            anime_df.loc[anime_df['anime_id'] == id, 'name'].values[0])
    return names


# Demander à l'utilisateur de saisir son ID
user_id = int(input("Entrez votre ID d'utilisateur : "))

# Obtenir les 10 meilleures recommendations pour l'utilisateur

top_10_movie_ids = get_top_10_recommendations(algo, user_id)

print(f"Voici la liste des films recommendé pour l'utilisateur {user_id}")
for nom in get_names(top_10_movie_ids):
    print(f"- {nom}")
