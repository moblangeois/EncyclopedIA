""""
title: Encyclopedia
author: Morgan Blangeois
version: 0.1
"""

# Standard library imports
import os
import logging
import pickle
import base64
import io
import traceback
import time
import uuid
import json
import requests
from collections import defaultdict
import re

# Third-party library imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any, Union
from openai import OpenAI
import tiktoken

# Local imports
from open_webui.utils.misc import get_last_user_message
from open_webui.apps.webui.models.files import Files


class Concepts(BaseModel):
    """Model for representing extracted concepts."""

    concepts: List[str] = Field(
        ..., description="List of key concepts extracted from the text"
    )


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="", description="Clé API OpenAI")
        GRAPH_FILE_URL: str = Field(
            default="https://raw.githubusercontent.com/moblangeois/EncyclopedIA/main/data/output_kg.jsonld",
            description="URL du fichier du graphe de connaissances JSON-LD",
        )
        MODEL_ID: str = Field(
            default="gpt-4o-mini", description="Modèle OpenAI à utiliser"
        )
        SIMILARITY_THRESHOLD: float = Field(
            default=0.6, description="Seuil de similarité pour ajouter des arêtes"
        )
        GRAPH_TEMPLATE_URL: str = Field(
            default="https://raw.githubusercontent.com/moblangeois/EncyclopedIA/main/graph_template.html",
            description="URL du template HTML pour le graphe",
        )

    def __init__(self):
        self.type = "pipe"
        self.valves = self.Valves()
        self.client = None
        self.knowledge_graph = None
        self.user_id = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def initialize(self):
        """Initialize OpenAI client and load knowledge graph."""
        self.client = OpenAI(api_key=self.valves.OPENAI_API_KEY)

        try:
            logging.info("Downloading knowledge graph from URL...")
            self.load_graph_from_url(self.valves.GRAPH_FILE_URL)

        except Exception as e:
            logging.error(f"Failed to load graph from URL: {str(e)}")
            raise

    def load_graph_from_url(self, url):
        """Download and load the knowledge graph from a given URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses

            if response.content:  # Vérifier que le contenu n'est pas vide
                jsonld_data = response.json()
            else:
                logging.error("Le fichier JSON-LD récupéré est vide")
                raise ValueError("Le fichier JSON-LD est vide ou invalide.")

            G = nx.Graph()

            for item in jsonld_data:
                if item["@type"] == "Article":
                    node_id = str(item.get("@id", "")).split("/")[
                        -1
                    ]  # Convert to string if it's a float
                    G.add_node(
                        node_id,
                        title=item.get("title", ""),
                        content=item.get("content", ""),
                        concepts=item.get("concepts", []),
                        embedding=item.get("embedding", None),
                    )
                # Utilisation des triples RDF pour créer les arêtes
                if "triples" in item:
                    for triple in item["triples"]:
                        # Convertir en chaîne de caractères avant d'appeler split()
                        source = str(triple.get("subject", "")).split("/")[-1]
                        target = str(triple.get("object", "")).split("/")[-1]
                        predicate = triple.get("predicate", "")
                        G.add_edge(source, target, label=predicate)

            self.knowledge_graph = G
            logging.info(
                f"Graphe chargé depuis URL. Nombre de nœuds : {G.number_of_nodes()}, Nombre d'arêtes : {G.number_of_edges()}"
            )
        except requests.RequestException as e:
            logging.error(
                f"Erreur lors de la récupération du fichier JSON-LD : {str(e)}"
            )
            raise
        except Exception as e:
            logging.error(f"Erreur lors du chargement du graphe : {str(e)}")
            raise

    def create_graph_html(self, graph_data):
        try:
            # Récupérer le template HTML depuis l'URL
            response = requests.get(self.valves.GRAPH_TEMPLATE_URL)
            response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
            html_content = response.text

            # Ajouter des styles CSS pour la hauteur
            style_to_add = """
            <style>
            html, body {
                height: 100%;
                margin: 0;
                padding: 0;
                overflow: hidden;
            }
            #graph-container {
                width: 100%;
                height: 100vh;  # Utilise toute la hauteur de la fenêtre
                position: relative;
            }
            #mynetwork {
                width: 100%;
                height: 100%;  # L'iframe prend toute la hauteur de son conteneur
                position: absolute;
                top: 0;
                left: 0;
            }
            </style>
            """

            # Insérer les styles juste avant la fermeture de la balise </head>
            html_content = html_content.replace("</head>", f"{style_to_add}</head>")

            # Remplacer les placeholders avec les données du graphe
            html_content = html_content.replace(
                "{{nodes}}", json.dumps(graph_data["nodes"])
            )
            html_content = html_content.replace(
                "{{edges}}", json.dumps(graph_data["edges"])
            )

            return html_content
        except requests.RequestException as e:
            logging.error(f"Erreur lors de la récupération du template HTML : {str(e)}")
            # Utiliser un template de secours simple en cas d'erreur
            return f"""
            <!DOCTYPE html>
            <html>
            <head><title>Graphe de parcours (version de secours)</title></head>
            <body>
                <pre>{json.dumps(graph_data, indent=2)}</pre>
            </body>
            </html>
            """

    def create_file(
        self, file_name: str, title: str, content: Union[str, bytes], content_type: str
    ):
        base_path = os.path.join(
            os.getcwd(), "EncyclopedIA"
        )  # Assurez-vous que ce chemin est correct
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        file_id = str(uuid.uuid4())
        file_path = os.path.join(base_path, f"{file_id}_{file_name}")

        # Créer le fichier
        mode = "w" if isinstance(content, str) else "wb"
        with open(file_path, mode) as f:
            f.write(content)

        meta = {
            "source": file_path,
            "title": title,
            "content_type": content_type,
            "size": os.path.getsize(file_path),
            "path": file_path,
        }

        class FileForm(BaseModel):
            id: str
            filename: str
            meta: dict = {}

        form_data = FileForm(id=file_id, filename=file_name, meta=meta)

        file = Files.insert_new_file(self.user_id, form_data)
        return file.id

    async def load_graph_from_jsonld(self, file_path, __event_emitter__=None):
        with open(file_path, "r", encoding="utf-8") as f:
            jsonld_data = json.load(f)

        G = nx.Graph()

        for item in jsonld_data:
            if item["@type"] == "Article":
                node_id = item["@id"].split("/")[-1]
                embedding = item.get("embedding", None)
                # Envoyer un message pour chaque nœud ingéré
                if __event_emitter__:
                    await __event_emitter__({"type": "message", "data": {"content": f"Nœud ajouté : {node_id}, avec embedding : {embedding}"}})
                G.add_node(
                    node_id,
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    concepts=item.get("concepts", []),
                    embedding=embedding,
                )
            # Utilisation des triples RDF pour créer les arêtes
            if "triples" in item:
                for triple in item["triples"]:
                    source = triple["subject"].split("/")[-1]
                    target = triple["object"].split("/")[-1]
                    predicate = triple["predicate"]
                    G.add_edge(source, target, label=predicate)

        self.knowledge_graph = G
        await __event_emitter__({
            "type": "message",
            "data": {"content": f"Graphe chargé avec {G.number_of_nodes()} nœuds et {G.number_of_edges()} arêtes."}
        })

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding

    def generate_concepts(self, text: str) -> List[str]:
        prompt = f"""
        Analysez le texte suivant provenant de l'Encyclopédie de Diderot et d'Alembert (18ème siècle) et identifiez les concepts clés. Votre tâche est de :
    
        1. Extraire les concepts principaux en tenant compte du contexte historique et philosophique de l'Encyclopédie.
        2. Prioriser les concepts qui reflètent les idées des Lumières, les avancées scientifiques et les débats philosophiques de l'époque.
        3. Inclure les termes spécifiques à la discipline traitée dans le texte (par exemple, philosophie, sciences naturelles, arts mécaniques, etc.).
        4. Identifier les concepts qui pourraient être controversés ou novateurs pour l'époque.
        5. Repérer les définitions importantes ou les tentatives de fixer le sens de certains termes.
        6. Noter les concepts qui font l'objet de renvois à d'autres articles de l'Encyclopédie.
        7. Être attentif aux nuances de sens et aux usages particuliers des termes dans ce contexte.
    
        Évitez les concepts redondants ou trop généraux. Limitez-vous à un maximum de 10 concepts clés, en privilégiant la pertinence et la spécificité par rapport au texte et à son contexte historique.
    
        Texte à analyser :
        {text}
    
        Listez les concepts clés identifiés :
        """

        parsed_response = self.client.beta.chat.completions.parse(
            model=self.valves.MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "Vous êtes un expert en histoire des idées du 18ème siècle, spécialisé dans l'Encyclopédie de Diderot et d'Alembert. Votre tâche est d'extraire les concepts clés des textes de l'Encyclopédie avec précision et perspicacité.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=Concepts,
        )

        return parsed_response.choices[0].message.parsed.concepts

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _retrieve_relevant_nodes(
        self, query_embedding: List[float], query: str, k: int = 10, min_similarity: float = 0.05, __event_emitter__=None
    ) -> List[Tuple[str, float]]:

        similarities = []
        for node in self.knowledge_graph.nodes:
            embedding = self.knowledge_graph.nodes[node].get("embedding", None)
            if embedding is not None:
                similarity = self._cosine_similarity(query_embedding, embedding)
                if similarity >= min_similarity:
                    similarities.append((node, similarity))
                # Envoyer des messages avec l'émetteur pour suivre la progression
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "message",
                        "data": {"content": f"Node: {node}, Similarity: {similarity}"}
                    })

        # Tri des résultats par similarité décroissante
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    def _expand_context(
        self, query: str, relevant_nodes: List[Tuple[str, float]]
    ) -> Tuple[str, List[str], Dict[str, str], str]:
        expanded_context = ""
        traversal_path = []
        filtered_content = {}

        query_concepts = self.generate_concepts(query)

        for node, similarity in relevant_nodes:
            if node not in traversal_path:
                traversal_path.append(node)
                node_data = self.knowledge_graph.nodes[node]
                node_content = node_data.get("content", "")
                node_concepts = node_data.get("concepts", [])
                node_author = node_data.get("author", "Auteur inconnu")
                node_domain = node_data.get("domain", "Domaine non spécifié")

                # Filtrer le contenu en fonction de la pertinence avec la requête
                relevant_sentences = self._extract_relevant_sentences(
                    node_content, query_concepts
                )
                filtered_content[node] = "\n".join(relevant_sentences)

                expanded_context += f"\n\nArticle : {node_data.get('title', '')}\n"
                expanded_context += f"Auteur : {node_author}\n"
                expanded_context += f"Domaine : {node_domain}\n"
                expanded_context += f"Concepts clés : {', '.join(node_concepts)}\n"
                expanded_context += f"Contenu pertinent : {filtered_content[node]}\n"
                expanded_context += f"Similarité avec la requête : {similarity:.2f}\n"

                # Étendre le contexte avec les nœuds liés
                neighbors = list(self.knowledge_graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in traversal_path:
                        traversal_path.append(neighbor)
                        neighbor_data = self.knowledge_graph.nodes[neighbor]
                        neighbor_content = neighbor_data.get("content", "")
                        neighbor_concepts = neighbor_data.get("concepts", [])

                        # Vérifier la pertinence du nœud voisin
                        if self._is_relevant(neighbor_concepts, query_concepts):
                            relevant_sentences = self._extract_relevant_sentences(
                                neighbor_content, query_concepts
                            )
                            filtered_content[neighbor] = "\n".join(relevant_sentences)

                            expanded_context += (
                                f"\n\nArticle lié : {neighbor_data.get('title', '')}\n"
                            )
                            expanded_context += f"Auteur : {neighbor_data.get('author', 'Auteur inconnu')}\n"
                            expanded_context += f"Domaine : {neighbor_data.get('domain', 'Domaine non spécifié')}\n"
                            expanded_context += (
                                f"Concepts clés : {', '.join(neighbor_concepts)}\n"
                            )
                            expanded_context += (
                                f"Contenu pertinent : {filtered_content[neighbor]}\n"
                            )

        prompt = f"""En vous basant sur le contexte suivant extrait de l'Encyclopédie de Diderot et d'Alembert, 
        veuillez répondre à la requête. Votre réponse doit :
        1. Synthétiser les informations pertinentes des différents articles.
        2. Mettre en évidence les liens entre les concepts et les idées des différents auteurs.
        3. Expliquer les éventuelles divergences ou évolutions dans la compréhension des concepts.
        4. Contextualiser la réponse dans le cadre des idées des Lumières du 18ème siècle.
        5. Mentionner explicitement les sources (articles et auteurs) utilisées dans votre réponse.
    
        Contexte : 
        {expanded_context}
    
        Requête : {query}
    
        Réponse :"""

        final_answer = (
            self.client.chat.completions.create(
                model=self.valves.MODEL_ID,
                messages=[
                    {
                        "role": "system",
                        "content": "Vous êtes un expert en histoire des idées du 18ème siècle, spécialisé dans l'analyse de l'Encyclopédie de Diderot et d'Alembert.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            .choices[0]
            .message.content
        )

        return expanded_context, traversal_path, filtered_content, final_answer

    def _extract_relevant_sentences(self, text: str, concepts: List[str]) -> List[str]:
        sentences = text.split(".")
        return [s for s in sentences if any(c.lower() in s.lower() for c in concepts)]

    def _is_relevant(self, node_concepts: List[str], query_concepts: List[str]) -> bool:
        return len(set(node_concepts) & set(query_concepts)) > 0

    def visualize_traversal(self, traversal_path: List[str]) -> Dict:
        G = self.knowledge_graph.subgraph(traversal_path)
        nodes = []
        edges = []

        for node in G.nodes():
            node_data = G.nodes[node]
            nodes.append(
                {
                    "id": node,
                    "label": node_data.get("title", node),
                    "content": node_data.get("content", ""),
                }
            )

        for edge in G.edges():
            edges.append({"from": edge[0], "to": edge[1]})

        return {"nodes": nodes, "edges": edges}

    async def query(
        self, query: str, __event_emitter__=None
    ) -> Tuple[str, List[str], Dict[str, str]]:

        query_embedding = self.get_embedding(query)

        await __event_emitter__(
            {
                "type": "message",
                "data": {"content": "Recherche des nœuds pertinents...\n"},
            }
        )
        relevant_nodes = self._retrieve_relevant_nodes(query_embedding, query)

        await __event_emitter__(
            {
                "type": "message",
                "data": {
                    "content": f"Nœuds pertinents trouvés : {[self.knowledge_graph.nodes[node].get('title', node) for node, _ in relevant_nodes]}\n"
                },
            }
        )

        await __event_emitter__(
            {
                "type": "message",
                "data": {"content": "Expansion du contexte...\n"},
            }
        )
        expanded_context, traversal_path, filtered_content, final_answer = (
            self._expand_context(query, relevant_nodes)
        )

        return final_answer, traversal_path, filtered_content

    async def pipe(self, body: dict, __user__: dict, __event_emitter__=None):
        if not self.client or not self.knowledge_graph:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Initialisation...", "done": False},
                }
            )
            self.initialize()

        self.user_id = __user__["id"]
        messages = body["messages"]
        user_message = get_last_user_message(messages)

        try:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Traitement de la requête...",
                        "done": False,
                    },
                }
            )

            answer, traversal_path, _ = await self.query(
                user_message, __event_emitter__
            )

            if answer is None or traversal_path is None:
                raise ValueError("La requête n'a pas produit de résultat valide.")

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Génération du graphe...", "done": False},
                }
            )
            graph_data = self.visualize_traversal(traversal_path)

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Création du fichier HTML du graphe...",
                        "done": False,
                    },
                }
            )

            html_content = self.create_graph_html(graph_data)

            # Créer un fichier HTML avec le graphe interactif
            graph_file_id = self.create_file(
                f"graph_{uuid.uuid4()}.html",
                "Graphe de parcours",
                html_content,
                "text/html",
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Création de la réponse...", "done": False},
                }
            )

            response = f"""Requête : {user_message}\nRéponse : {answer}\nGraphe de parcours :\n\n{{{{HTML_FILE_ID_{graph_file_id}}}}}
            """

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Traitement terminé", "done": True},
                }
            )
            return response

        except Exception as e:
            error_msg = f"Erreur lors du traitement de la requête : {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Une erreur s'est produite : {str(e)}",
                        "done": True,
                    },
                }
            )
            return error_msg
