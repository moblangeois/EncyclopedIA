import logging
import pickle
import json
import requests
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pydantic import BaseModel
from typing import List
import tiktoken
import os
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

class Concepts(BaseModel):
    concepts: List[str]

class KnowledgeGraph:
    def __init__(self, similarity_threshold=0.6, openai_api_key=None):
        self.graph = nx.Graph()  # Initialisation d'un graphe non orienté
        self.similarity_threshold = similarity_threshold  # Seuil de similarité pour ajouter des arêtes
        self.df = None  # Initialisation d'un DataFrame pour stocker les données
        self.openai_api_key = openai_api_key
        self.headers = {"Authorization": f"Bearer {openai_api_key}"} if openai_api_key else {}
        self.client = OpenAI(api_key=openai_api_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Utilisez l'encodeur approprié pour votre modèle

    def load_data(self, file_path):
        """
        Charge les données à partir d'un fichier TSV.
        
        Parameters
        ----------
        file_path : str
            Chemin vers le fichier TSV contenant les données.
        """
        logging.info("Chargement des données...")  # Chargement des données à partir d'un fichier TSV
        self.df = pd.read_csv(file_path, sep='\t')  # Lecture du fichier TSV
        self.df['content'] = self.df['content'].astype(str).replace('nan', '')  # Remplacement des valeurs NaN par des chaînes vides
        self.df['head'] = self.df['head'].astype(str).replace('nan', '')  # Remplacement des valeurs NaN par des chaînes vides
        self.df = self.df[self.df['content'].str.strip() != '']  # Suppression des lignes avec un contenu vide
        logging.info(f"Données chargées. Nombre d'articles : {len(self.df)}")  # Affichage du nombre d'articles chargés

    def create_embeddings(self):
        """
        Crée les embeddings pour chaque article en utilisant l'API OLLAMA.
        """
        logging.info("Création des embeddings...")  # Création des embeddings pour chaque article en utilisant l'API OLLAMA
        tqdm.pandas()  # Affichage d'une barre de progression
        self.df['embedding'] = self.df['content'].progress_apply(lambda x: self.get_embedding(x))  # Application de la fonction get_embedding à chaque ligne
        logging.info("Embeddings créés.")  # Affichage d'un message de confirmation

    def create_embeddings_and_concepts(self):
        logging.info("Création des embeddings et des concepts...")
        tqdm.pandas()
        
        embeddings_file = 'data/embeddings.pkl'
        if os.path.exists(embeddings_file):
            self.load_embeddings(embeddings_file)
        else:
            self.df['embedding'] = self.df['content'].progress_apply(lambda x: self.get_embedding(x))
            self.save_embeddings(embeddings_file)
        
        self.df['concepts'] = self.df['content'].progress_apply(lambda x: self.generate_concepts(x))
        logging.info("Embeddings et concepts créés.")

    def save_embeddings(self, file_path):
        embeddings = self.df['embedding'].tolist()
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings sauvegardés dans {file_path}")

    def load_embeddings(self, file_path):
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        self.df['embedding'] = embeddings
        print(f"Embeddings chargés depuis {file_path}")
        
    def generate_concepts(self, text):
        url = "https://api.openai.com/v1/chat/completions"
        prompt = f"""
        L'objectif est d'identifier les concepts clés présents dans le texte suivant afin de comprendre les informations essentielles qu'il contient.
        Dans le cadre de cette analyse, vous devez extraire les concepts principaux du texte ci-dessous.
        TRÈS IMPORTANT : Ne générez pas de concepts redondants ou qui se chevauchent. Par exemple, si le texte contient les concepts "entreprise" et "société", vous ne devez en retenir qu'un seul.
        Ne vous souciez pas de la quantité, privilégiez toujours la qualité à la quantité. Assurez-vous que CHAQUE concept dans votre réponse est pertinent par rapport au contexte du texte.
        Et rappelez-vous, ce sont des CONCEPTS CLÉS que nous recherchons.
        Retournez les concepts sous forme d'une liste de chaînes de caractères.

        Texte : {text}
        """
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "Vous êtes un assistant spécialisé dans l'extraction de concepts clés."},
                    {"role": "user", "content": prompt},
                ],
                response_format=Concepts,
            )

            return completion.choices[0].message.parsed.concepts
        except Exception as e:
            raise Exception(f"Erreur lors de la génération des concepts: {str(e)}")

    def get_embedding(self, text):
        max_tokens = 8000  # Laissez une marge pour être sûr
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= max_tokens:
            return self._get_embedding_for_text(text)
        else:
            # Diviser le texte en morceaux
            chunks = []
            current_chunk = []
            current_length = 0
            for token in tokens:
                if current_length + 1 > max_tokens:
                    chunks.append(self.tokenizer.decode(current_chunk))
                    current_chunk = [token]
                    current_length = 1
                else:
                    current_chunk.append(token)
                    current_length += 1
            if current_chunk:
                chunks.append(self.tokenizer.decode(current_chunk))
            
            # Obtenir l'embedding pour chaque morceau
            embeddings = [self._get_embedding_for_text(chunk) for chunk in chunks]
            
            # Faire la moyenne des embeddings
            avg_embedding = [sum(e) / len(e) for e in zip(*embeddings)]
            return avg_embedding

    def _get_embedding_for_text(self, text):
        url = "https://api.openai.com/v1/embeddings"
        payload = {
            "input": text,
            "model": "text-embedding-3-small"  # ou "text-embedding-3-large"
        }
        response = requests.post(url, json=payload, headers=self.headers)
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        else:
            raise Exception(f"Erreur lors de la génération de l'embedding: {response.text}")

    def build_graph(self):
        """
        Construit le graphe de connaissances en ajoutant des nœuds et des arêtes.
        """
        logging.info("Construction du graphe...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Ajout des nœuds"):
            node_id = str(row['id_enccre'])
            node_data = row.to_dict()
            node_data['concepts'] = row['concepts']
            self.graph.add_node(node_id, **node_data)

        embeddings = np.array(self.df['embedding'].tolist())
        similarity_matrix = cosine_similarity(embeddings)

        for i in tqdm(range(len(self.df)), desc="Ajout des arêtes"):
            for j in range(i+1, len(self.df)):
                embedding_similarity = similarity_matrix[i][j]
                concept_similarity = self._calculate_concept_similarity(self.df.iloc[i]['concepts'], self.df.iloc[j]['concepts'])
                
                combined_similarity = 0.5 * embedding_similarity + 0.5 * concept_similarity
                
                if combined_similarity > self.similarity_threshold:
                    self.graph.add_edge(str(self.df.iloc[i]['id_enccre']), str(self.df.iloc[j]['id_enccre']), 
                                        weight=combined_similarity)

        logging.info(f"Graphe construit. Nombre de nœuds : {self.graph.number_of_nodes()}, Nombre d'arêtes : {self.graph.number_of_edges()}")

    def _calculate_concept_similarity(self, concepts1, concepts2):
        """
        Calcule la similarité entre deux ensembles de concepts.
        
        Parameters
        ----------
        concepts1 : list
            Liste de concepts du premier ensemble.
        concepts2 : list
            Liste de concepts du deuxième ensemble.
        
        Returns
        -------
        float
            Similarité entre les deux ensembles de concepts.
        """
        set1 = set(concepts1)
        set2 = set(concepts2)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def __getstate__(self):
        # Cette méthode est appelée lors de la sérialisation
        state = self.__dict__.copy()
        # Supprimez le client OpenAI du dictionnaire d'état
        del state['client']
        return state

    def __setstate__(self, state):
        # Cette méthode est appelée lors de la désérialisation
        self.__dict__.update(state)
        # Recréez le client OpenAI
        self.client = OpenAI(api_key=self.openai_api_key)

    def save_graph(self, file_path):
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            
            # Vérification d'intégrité
            with open(file_path, 'rb') as f:
                _ = pickle.load(f)
            
            logging.info(f"Graphe sauvegardé et vérifié dans {file_path}")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde ou de la vérification du graphe: {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise

    @classmethod
    def load_graph(cls, file_path, openai_api_key):
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)
        # Assurez-vous que le client OpenAI est recréé après le chargement
        graph.openai_api_key = openai_api_key
        graph.client = OpenAI(api_key=openai_api_key)
        return graph

    def get_node_info(self, node_id):
        """
        Retourne les informations d'un nœud donné.
        
        Parameters
        ----------
        node_id : str
            ID du nœud.
        
        Returns
        -------
        dict
            Informations du nœud.
        """
        return self.graph.nodes[node_id]  # Retourne les informations d'un nœud

    def get_neighbors(self, node_id):
        """
        Retourne une liste des voisins d'un nœud donné.
        
        Parameters
        ----------
        node_id : str
            ID du nœud.
        
        Returns
        -------
        list
            Liste des voisins du nœud.
        """
        return list(self.graph.neighbors(node_id))  # Retourne une liste des voisins d'un nœud

    def get_subgraph(self, node_ids):
        """
        Retourne un sous-graphe contenant les nœuds spécifiés.
        
        Parameters
        ----------
        node_ids : list
            Liste des IDs des nœuds à inclure dans le sous-graphe.
        
        Returns
        -------
        nx.Graph
            Sous-graphe contenant les nœuds spécifiés.
        """
        return self.graph.subgraph(node_ids)  # Retourne un sous-graphe contenant les nœuds spécifiés

    def get_most_connected_nodes(self, n=10):
        """
        Retourne les nœuds les plus connectés selon le degré.
        
        Parameters
        ----------
        n : int, optional
            Nombre de nœuds à retourner (default is 10).
        
        Returns
        -------
        list
            Liste des nœuds les plus connectés.
        """
        return sorted(self.graph.degree, key=lambda x: x[1], reverse=True)[:n]  # Retourne les nœuds les plus connectés selon le degré (nombre d'arêtes)

    def get_shortest_path(self, start_node, end_node):
        """
        Retourne le chemin le plus court entre deux nœuds.
        
        Parameters
        ----------
        start_node : str
            ID du nœud de départ.
        end_node : str
            ID du nœud d'arrivée.
        
        Returns
        -------
        list
            Liste des nœuds formant le chemin le plus court.
        """
        return nx.shortest_path(self.graph, start_node, end_node)

    def get_connected_components(self):
        """
        Retourne une liste de composants connexes.

        Returns
        -------
        list
            Liste de composants connexes.
        """

        return list(nx.connected_components(self.graph)) # Retourne une liste de composants connexes

    def visualize_graph(self, output_file='graph.png'):
        """
        Visualise le graphe de connaissances et sauvegarde l'image.
        
        :param output_file: Nom du fichier image de sortie
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        clusters = nx.get_node_attributes(self.graph, 'cluster')
        colors = [plt.cm.jet(float(clusters[node]) / max(clusters.values())) for node in self.graph.nodes()]
        
        nx.draw(self.graph, pos, with_labels=True, node_color=colors, 
                node_size=500, font_size=8, font_weight='bold')
        
        # Ajout des titres comme labels
        labels = nx.get_node_attributes(self.graph, 'title')
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=6)
        
        plt.title("Graphe de connaissances")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graphe sauvegardé dans {output_file}")

    def visualize_subgraph(self, node_ids, output_file='subgraph.png'):
        """
        Visualise un sous-graphe contenant les nœuds spécifiés et sauvegarde l'image.

        :param node_ids: Liste des IDs des nœuds à inclure dans le sous-graphe
        :param output_file: Nom du fichier image de sortie
        """
        subgraph = self.get_subgraph(node_ids)
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8)
        
        labels = nx.get_node_attributes(subgraph, 'title')
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=6)
        
        plt.title("Sous-graphe")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sous-graphe sauvegardé dans {output_file}")

    def _lemmatize_concept(self, concept):
        # Cette méthode devrait peut-être être implémentée pour lemmatiser les concepts
        # Pour l'instant, nous retournons simplement le concept en minuscules
        return concept.lower()