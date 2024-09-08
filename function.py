"""
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
import asyncio

# Third-party library imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any, Union, Optional
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
        GRAPH_TEMPLATE_URL: str = Field(
            default="https://raw.githubusercontent.com/moblangeois/EncyclopedIA/main/graph_template.html",
            description="URL du template HTML pour le graphe",
        )
        MAX_DEPTH: int = Field(
            default=2, description="Profondeur maximale pour suivre les renvois"
        )
        TOP_K_NODES: int = Field(
            default=10, description="Nombre de nœuds les plus pertinents à considérer"
        )
        MIN_SIMILARITY: float = Field(
            default=0.4, description="Similarité minimale pour considérer un nœud"
        )
        MAX_ARTICLE_WORDS: int = Field(
            default=400,
            description="Nombre maximum de mots pour un article avant résumé",
        )
        QUERY_TIMEOUT: int = Field(
            default=60,
            description="Temps maximum en secondes pour traiter une requête",
        )

    def __init__(self):
        self.type = "pipe"
        self.valves = self.Valves()
        self.client = None
        self.knowledge_graph = None
        self.user_id = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def initialize(self, __event_emitter__=None):
        """Initialize OpenAI client and load knowledge graph."""
        try:
            if self.valves.OPENAI_API_KEY:
                self.client = OpenAI(api_key=self.valves.OPENAI_API_KEY)
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": "Client OpenAI initialisé avec succès.\n"
                            },
                        }
                    )
            else:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": "Erreur : Clé API OpenAI manquante..\n"
                            },
                        }
                    )
                raise ValueError("La clé API OpenAI n'est pas définie..\n")

            # Check for local file first
            local_graph_path = "/app/backend/EDdA_knowledge_graph.jsonld"
            if os.path.exists(local_graph_path):
                await self.load_graph_from_jsonld(local_graph_path, __event_emitter__)
            else:
                # If local file doesn't exist, download the graph from the URL
                await self.load_graph_from_url(
                    self.valves.GRAPH_FILE_URL, __event_emitter__
                )

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": f"Erreur lors de l'initialisation : {str(e)}.\n"
                        },
                    }
                )
            raise

    async def load_graph_from_url(self, url, __event_emitter__=None):
        """Download and load the knowledge graph from a given URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses

            if response.content:  # Vérifier que le contenu n'est pas vide
                jsonld_data = response.json()
            else:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": "Le fichier JSON-LD récupéré est vide"},
                        }
                    )
                raise ValueError("Le fichier JSON-LD est vide ou invalide.")

            G = nx.Graph()

            for item in jsonld_data:
                if item["@type"] == "Article":
                    # Assurez-vous que l'ID n'est pas vide et que le noeud est bien ajouté avec un ID valide
                    node_id = (
                        str(item.get("@id", "")).split("/")[-1]
                        if item.get("@id")
                        else None
                    )
                    if node_id:
                        embedding = item.get("embedding", None)

                        G.add_node(
                            node_id,
                            title=item.get("title", ""),
                            content=item.get("content", ""),
                            concepts=item.get("concepts", []),
                            embedding=embedding,
                        )
                    else:
                        # Log si l'ID est absent ou vide
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "message",
                                    "data": {
                                        "content": f"Erreur : Noeud sans ID valide trouvé. Article : {item.get('title', 'Sans titre')}"
                                    },
                                }
                            )

            self.knowledge_graph = G
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": f"Graphe chargé depuis URL avec {G.number_of_nodes()} nœuds et {G.number_of_edges()} arêtes.\n"
                        },
                    }
                )

        except requests.RequestException as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": f"Erreur lors de la récupération du fichier JSON-LD : {str(e)}"
                        },
                    }
                )
            raise
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": f"Erreur lors du chargement du graphe : {str(e)}"
                        },
                    }
                )
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

    def _find_node_by_title(self, title: str) -> Optional[str]:
        for node, data in self.knowledge_graph.nodes(data=True):
            if data.get("title", "").lower() == title.lower():
                return node
        return None

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
                # Ensure @id is a string and check its type before splitting
                raw_id = item.get("@id", "")
                if isinstance(raw_id, str):
                    node_id = raw_id.split("/")[-1]
                else:
                    # Handle non-string @id (e.g., floats or other types)
                    """
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "message",
                                "data": {"content": f"Skipping invalid ID: {raw_id}"},
                            }
                        )
                    """
                    continue  # Skip this item if the ID is not valid

                embedding = item.get("embedding", None)
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
                    subject = triple.get("subject", "")
                    object_ = triple.get("object", "")

                    # Ensure both subject and object are strings before splitting
                    if isinstance(subject, str) and isinstance(object_, str):
                        source = subject.split("/")[-1]
                        target = object_.split("/")[-1]
                        predicate = triple.get("predicate", "")
                        G.add_edge(source, target, label=predicate)
                    else:
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "message",
                                    "data": {
                                        "content": f"Skipping invalid triple: subject={subject}, object={object_}"
                                    },
                                }
                            )
                        continue  # Skip invalid triples

        self.knowledge_graph = G
        await __event_emitter__(
            {
                "type": "message",
                "data": {
                    "content": f"Graphe chargé avec {G.number_of_nodes()} nœuds et {G.number_of_edges()} arêtes."
                },
            }
        )

    def get_embedding(self, text: str) -> List[float]:
        if not self.client:
            raise ValueError("Le client OpenAI n'est pas initialisé.")

        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Erreur lors de la génération d'embedding : {str(e)}")
            raise

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

    async def _retrieve_relevant_nodes(
        self,
        query_embedding: List[float],
        query: str,
        __event_emitter__=None,
    ) -> List[Tuple[str, float]]:
        similarities = []
        for node in self.knowledge_graph.nodes:
            embedding = self.knowledge_graph.nodes[node].get("embedding", None)
            if embedding is not None:
                similarity = self._cosine_similarity(query_embedding, embedding)
                if similarity >= self.valves.MIN_SIMILARITY:
                    similarities.append((node, similarity))

        # Tri des résultats par similarité décroissante
        return sorted(similarities, key=lambda x: x[1], reverse=True)[
            : self.valves.TOP_K_NODES
        ]

    async def _expand_context(
        self,
        query: str,
        relevant_nodes: List[Tuple[str, float]],
        __event_emitter__=None,
    ) -> Tuple[str, List[str], Dict[str, str], str]:
        expanded_context = ""
        traversal_path = []
        filtered_content = {}
        references_explored = set()  # Pour éviter les boucles infinies

        query_concepts = self.generate_concepts(query)

        async def explore_node(node, depth, similarity=None, from_reference=False):
            if depth > self.valves.MAX_DEPTH or node in references_explored:
                return

            references_explored.add(node)
            if node not in traversal_path:
                traversal_path.append(node)
                node_data = self.knowledge_graph.nodes[node]
                node_content = node_data.get("content", "")
                node_concepts = node_data.get("concepts", [])
                node_author = node_data.get("author", "Auteur inconnu")
                node_domain = node_data.get("domain", "Domaine non spécifié")
                node_references = node_data.get("references", [])

                prepared_content = await self._prepare_article_content(node_content)
                filtered_content[node] = prepared_content

                nonlocal expanded_context
                expanded_context += f"\n\n{'='*50}\n"
                expanded_context += f"Article : {node_data.get('title', '')}\n"
                expanded_context += f"Auteur : {node_author}\n"
                expanded_context += f"Domaine : {node_domain}\n"
                expanded_context += f"Concepts clés : {', '.join(node_concepts)}\n"
                expanded_context += f"Contenu : {filtered_content[node]}\n"
                if similarity is not None:
                    expanded_context += (
                        f"Similarité avec la requête : {similarity:.2f}\n"
                    )
                if from_reference:
                    expanded_context += (
                        "Cet article a été exploré à partir d'un renvoi.\n"
                    )

                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": f"Exploration de l'article : {node_data.get('title', '')}\n"
                        },
                    }
                )

                # Explorer les renvois
                for ref in node_references:
                    ref_node = self._find_node_by_title(ref)
                    if ref_node:
                        await __event_emitter__(
                            {
                                "type": "message",
                                "data": {
                                    "content": f"Suivant le renvoi vers : {ref}\n"
                                },
                            }
                        )
                        await explore_node(ref_node, depth + 1, from_reference=True)

                # Explorer les nœuds voisins
                neighbors = list(self.knowledge_graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in traversal_path:
                        neighbor_data = self.knowledge_graph.nodes[neighbor]
                        neighbor_concepts = neighbor_data.get("concepts", [])
                        if self._is_relevant(neighbor_concepts, query_concepts):
                            await __event_emitter__(
                                {
                                    "type": "message",
                                    "data": {
                                        "content": f"Exploration du nœud voisin : {neighbor_data.get('title', '')}\n"
                                    },
                                }
                            )
                            await explore_node(neighbor, depth + 1)

        for node, similarity in relevant_nodes:
            await explore_node(node, 0, similarity)

        prompt = f"""En vous basant sur le contexte suivant extrait de l'Encyclopédie de Diderot et d'Alembert, 
        veuillez répondre à la requête. Votre réponse doit :
        1. Synthétiser les informations pertinentes des différents articles.
        2. Mettre en évidence les liens entre les concepts et les idées des différents auteurs.
        3. Expliquer les éventuelles divergences ou évolutions dans la compréhension des concepts.
        4. Contextualiser la réponse dans le cadre des idées des Lumières du 18ème siècle.
        5. Mentionner explicitement les sources (articles et auteurs) utilisées dans votre réponse.
        6. Indiquer les renvois pertinents et expliquer leur importance dans le contexte de la requête.
        7. Ne s'appuyer que sur le contexte fourni.
    
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
                        "content": "Vous vous appuyez sur le contexte fourni pour rédiger votre réponse.",
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

    async def _prepare_article_content(self, content: str) -> str:
        words = content.split()
        if len(words) <= self.valves.MAX_ARTICLE_WORDS:
            return content
        else:
            # Utiliser l'API pour résumer l'article
            summary_prompt = f"""Résumez l'article suivant de l'Encyclopédie en environ {self.valves.MAX_ARTICLE_WORDS} mots, 
            en conservant les informations et concepts clés :
    
            {content}
    
            Résumé :"""

            summary = (
                self.client.chat.completions.create(
                    model=self.valves.MODEL_ID,
                    messages=[
                        {
                            "role": "system",
                            "content": "Vous êtes un expert en résumé de textes historiques.",
                        },
                        {"role": "user", "content": summary_prompt},
                    ],
                )
                .choices[0]
                .message.content
            )

            return summary

    def _is_relevant(self, node_concepts: List[str], query_concepts: List[str]) -> bool:
        return len(set(node_concepts) & set(query_concepts)) > 0

    def visualize_traversal(self, traversal_path: List[str]) -> Dict:
        G = self.knowledge_graph.subgraph(traversal_path)
        nodes = []
        edges = []

        for i, node in enumerate(traversal_path):
            node_data = G.nodes[node]
            nodes.append(
                {
                    "id": node,
                    "label": f"{i+1}. {node_data.get('title', node)}",
                    "content": node_data.get("content", ""),
                    "references": node_data.get("references", []),
                    "order": i,
                }
            )

        for edge in G.edges():
            edges.append({"from": edge[0], "to": edge[1]})

        # Ajouter des arêtes pour les renvois
        for i, node in enumerate(traversal_path):
            node_data = G.nodes[node]
            for ref in node_data.get("references", []):
                ref_node = self._find_node_by_title(ref)
                if ref_node and ref_node in traversal_path:
                    edges.append(
                        {
                            "from": node,
                            "to": ref_node,
                            "dashes": True,
                            "label": f"Renvoi {i+1} -> {traversal_path.index(ref_node)+1}",
                            "arrows": "to",
                        }
                    )

        return {"nodes": nodes, "edges": edges}

    async def query(
        self, query: str, __event_emitter__=None
    ) -> Tuple[str, List[str], Dict[str, str]]:
        try:
            query_embedding = self.get_embedding(query)

            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": "Recherche des nœuds pertinents...\n"},
                }
            )

            relevant_nodes = await self._retrieve_relevant_nodes(
                query_embedding, query, __event_emitter__=__event_emitter__
            )

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
                    "data": {
                        "content": f"Expansion du contexte et exploration des renvois (profondeur max : {self.valves.MAX_DEPTH})...\n"
                    },
                }
            )

            expanded_context, traversal_path, filtered_content, final_answer = (
                await self._expand_context(query, relevant_nodes, __event_emitter__)
            )

            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": "Génération de la réponse finale...\n"},
                }
            )

            return final_answer, traversal_path, filtered_content

        except Exception as e:
            logging.error(f"Erreur dans la méthode query : {str(e)}")
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": f"Erreur dans la méthode query : {str(e)}\n"},
                }
            )
            raise

    async def pipe(self, body: dict, __user__: dict, __event_emitter__=None):
        if not self.client or not self.knowledge_graph:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Initialisation...", "done": False},
                }
            )
            await self.initialize(__event_emitter__)

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

            # Utiliser asyncio.wait_for pour implémenter un timeout
            try:
                answer, traversal_path, _ = await asyncio.wait_for(
                    self.query(user_message, __event_emitter__),
                    timeout=self.valves.QUERY_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Le traitement de la requête a dépassé le temps imparti ({self.valves.QUERY_TIMEOUT} secondes)"
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
