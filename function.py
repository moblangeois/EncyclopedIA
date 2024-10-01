"""
title: Encycloscope
author: Morgan Blangeois
version: 0.8
requirements: openai>=1.44.1
"""

# Standard library imports
import os
import logging
import uuid
import json
import asyncio

# Third-party library imports
import networkx as nx
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Union, Optional
from openai import OpenAI
import tiktoken

# Local imports
from open_webui.utils.misc import get_last_user_message
from open_webui.apps.webui.models.files import Files


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="", description="Clé API OpenAI")
        GRAPH_FILE_PATH: str = Field(
            default="data/EDdA_knowledge_graph_sample.jsonld",
            description="Chemin local du fichier du graphe de connaissances JSON-LD",
        )
        GRAPH_TEMPLATE_PATH: str = Field(
            default="data/graph_template.html",
            description="Chemin local du fichier HTML pour le graphe",
        )
        MODEL_ID: str = Field(
            default="gpt-4o-mini", description="Modèle OpenAI à utiliser"
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
                raise ValueError("La clé API OpenAI n'est pas définie.\n")

            # Récupérer les chemins des fichiers depuis les valves
            local_graph_path = self.valves.GRAPH_FILE_PATH
            local_html_template_path = self.valves.GRAPH_TEMPLATE_PATH

            # Charger le graphe depuis le fichier JSON-LD
            if os.path.exists(local_graph_path):
                await self.load_graph_from_jsonld(local_graph_path, __event_emitter__)
            else:
                raise FileNotFoundError(
                    f"Le fichier JSON-LD {local_graph_path} est introuvable."
                )

            # Charger le template HTML pour le graphe
            if os.path.exists(local_html_template_path):
                with open(local_html_template_path, "r", encoding="utf-8") as f:
                    self.html_template_content = f.read()
            else:
                raise FileNotFoundError(
                    f"Le fichier HTML {local_html_template_path} est introuvable."
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

    def create_graph_html(self, graph_data):
        try:
            # Utiliser le template HTML chargé depuis le fichier local
            html_content = self.html_template_content

            html_content = html_content.replace(
                "{{nodes}}", json.dumps(graph_data["nodes"], ensure_ascii=False)
            )
            html_content = html_content.replace(
                "{{edges}}", json.dumps(graph_data["edges"], ensure_ascii=False)
            )

            return html_content

        except Exception as e:
            logging.error(f"Erreur lors de la création du fichier HTML : {str(e)}")
            return f"<html><body><pre>Erreur : {str(e)}</pre></body></html>"

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
                    embedding=item.get("embedding", None),
                    url=item.get("url", "URL non disponible"),
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
                    "content": f"Graphe chargé avec {G.number_of_nodes()} nœuds et {G.number_of_edges()} arêtes.\n"
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

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            raise ValueError(
                f"Les dimensions des embeddings ne correspondent pas : {len(a)} != {len(b)}"
            )

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
    ) -> Tuple[str, List[str], Dict[str, str], str, Dict[str, str]]:
        expanded_context = ""
        traversal_path = []
        filtered_content = {}
        references_explored = set()  # Pour éviter les boucles infinies
        node_urls = {}  # Dictionnaire pour stocker les URLs des nœuds utilisés

        async def explore_node(node, depth, similarity=None, from_reference=False):
            # S'assurer que l'on respecte la profondeur maximale définie
            if depth > self.valves.MAX_DEPTH or node in references_explored:
                return

            references_explored.add(
                node
            )  # Ajouter le nœud exploré à la liste des nœuds déjà visités

            if node not in traversal_path:
                traversal_path.append(node)
                node_data = self.knowledge_graph.nodes[node]
                node_content = node_data.get("content", "")
                node_author = node_data.get("author", "Auteur inconnu")
                node_domain = node_data.get("domain", "Domaine non spécifié")
                node_references = node_data.get("references", [])
                node_url = node_data.get("url", "URL non disponible")

                # Ajouter l'URL de l'article au dictionnaire
                node_urls[node_data.get("title", node)] = node_url

                prepared_content = await self._prepare_article_content(node_content)
                filtered_content[node] = prepared_content

                nonlocal expanded_context
                expanded_context += f"\n\n{'='*50}\n"
                expanded_context += f"Article : {node_data.get('title', '')}\n"
                expanded_context += f"Auteur : {node_author}\n"
                expanded_context += f"Domaine : {node_domain}\n"
                expanded_context += f"Contenu : {filtered_content[node]}\n"
                expanded_context += (
                    f"URL : {node_url}\n"  # Ajout de l'URL dans le contexte
                )
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
                    if ref_node and ref_node not in references_explored:
                        await __event_emitter__(
                            {
                                "type": "message",
                                "data": {
                                    "content": f"Suivant le renvoi vers : {ref}\n"
                                },
                            }
                        )
                        await explore_node(ref_node, depth + 1, from_reference=True)

                # Explorer les nœuds voisins uniquement si la profondeur n'a pas été dépassée
                if depth < self.valves.MAX_DEPTH:
                    neighbors = list(self.knowledge_graph.neighbors(node))
                    for neighbor in neighbors:
                        if (
                            neighbor not in traversal_path
                            and neighbor not in references_explored
                        ):
                            await explore_node(neighbor, depth + 1)

        for node, similarity in relevant_nodes:
            await explore_node(node, 0, similarity)

        prompt = f"""En vous basant sur le contexte suivant extrait de l'Encyclopédie de Diderot et d'Alembert, 
        veuillez répondre à la requête. Votre réponse doit :
        1. Synthétiser les informations pertinentes des différents articles.
        2. Mettre en évidence les liens entre les concepts et les idées des différents auteurs.
        3. Expliquer les éventuelles divergences ou évolutions dans la compréhension des concepts.
        4. Contextualiser la réponse dans le cadre des idées des Lumières du 18ème siècle.
        5. Mentionner explicitement les sources (articles et auteurs) utilisées dans votre réponse, mais pas dans une partie à part.
        6. Indiquer les renvois pertinents et expliquer leur importance dans le contexte de la requête.
        7. Ne s'appuyer que sur le contexte fourni.
    
        Il faut répondre en suivant la structure suivante uniquement :
    
        ### Synthèse des articles
    
        [Synthèse]
    
        ### Analyse des concepts
    
        [Analyse]
    
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

        return (
            expanded_context,
            traversal_path,
            filtered_content,
            final_answer,
            node_urls,
        )

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
                            "content": "Vous êtes un expert en résumé de textes historiques. Répondez avec le texte résumé, sans rien d'autre",
                        },
                        {"role": "user", "content": summary_prompt},
                    ],
                )
                .choices[0]
                .message.content
            )

            return summary

    def visualize_traversal(self, traversal_path: List[str]) -> Dict:
        G = self.knowledge_graph.subgraph(traversal_path)
        nodes = []
        edges = []

        if not traversal_path:
            logging.error("Le chemin de parcours est vide.")
            return {"nodes": [], "edges": []}  # Retourner des tableaux vides

        for i, node in enumerate(traversal_path):
            node_data = G.nodes[node]
            nodes.append(
                {
                    "id": node,
                    "label": f"{i+1}. {node_data.get('title', node)}",
                    "content": node_data.get("content", "Pas de contenu disponible"),
                    "references": node_data.get("references", []),
                    "order": i,  # Ordre pour surligner le chemin
                }
            )

        for edge in G.edges():
            edges.append({"from": edge[0], "to": edge[1]})

        return {"nodes": nodes, "edges": edges}

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
    ) -> Tuple[str, List[str], Dict[str, str], str, Dict[str, str]]:
        try:
            query_embedding = self.get_embedding(query)

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

            (
                expanded_context,
                traversal_path,
                filtered_content,
                final_answer,
                node_urls,
            ) = await self._expand_context(query, relevant_nodes, __event_emitter__)

            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": "Génération de la réponse finale...\n"},
                }
            )

            return final_answer, traversal_path, filtered_content, node_urls

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

            try:
                answer, traversal_path, filtered_content, node_urls = (
                    await asyncio.wait_for(
                        self.query(user_message, __event_emitter__),
                        timeout=self.valves.QUERY_TIMEOUT,
                    )
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

            # Construction des liens URL à inclure dans la réponse finale
            urls_section = "\n\n### URLs des articles utilisés :\n"
            for title, url in node_urls.items():
                urls_section += f"- [{title}]({url})\n"

            response = f"""{answer}\n\n### Graphe de parcours \n\n{{{{HTML_FILE_ID_{graph_file_id}}}}}\n{urls_section}
            """

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Traitement terminé", "done": True},
                }
            )
            return response

        except Exception as e:
            error_msg = f"Erreur lors du traitement de la requête : {str(e)}"
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
