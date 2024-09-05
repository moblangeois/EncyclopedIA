import requests
import json
import heapq
import numpy as np
from typing import Tuple, List, Dict
from openai import OpenAI
from pydantic import BaseModel


class Concepts(BaseModel):
    concepts: List[str]

class QueryEngine:
    def __init__(self, knowledge_graph, openai_api_key):
        self.knowledge_graph = knowledge_graph
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        self.max_context_length = 7000


    def get_embedding(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding


    def generate_concepts(self, text):
        prompt = f"""
        L'objectif est d'identifier les concepts clés présents dans le texte suivant afin de comprendre les informations essentielles qu'il contient.
        Dans le cadre de cette analyse, vous devez extraire les concepts principaux du texte ci-dessous.
        TRÈS IMPORTANT : Ne générez pas de concepts redondants ou qui se chevauchent. Par exemple, si le texte contient les concepts "entreprise" et "société", vous ne devez en retenir qu'un seul.
        Ne vous souciez pas de la quantité, privilégiez toujours la qualité à la quantité. Assurez-vous que CHAQUE concept dans votre réponse est pertinent par rapport au contexte du texte.
        Et rappelez-vous, ce sont des CONCEPTS CLÉS que nous recherchons.

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

    def _calculate_concept_similarity(self, concepts1, concepts2):
        # Calculer la similarité entre deux ensembles de concepts
        # Par exemple, utiliser le coefficient de Jaccard
        set1 = set(concepts1)
        set2 = set(concepts2)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def generate_response(self, prompt):
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini-2024-07-18",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(url, json=payload, headers=self.headers)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Erreur lors de la génération de la réponse: {response.text}")

    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
        prompt = f"Étant donné la requête : '{query}'\n\nEt le contexte actuel :\n{context}\n\nCe contexte fournit-il une réponse complète à la requête ? Si oui, fournissez la réponse. Si non, indiquez que la réponse est incomplète.\n\nRéponse complète (Oui/Non) :\nRéponse (si complète) :"
        response = self.generate_response(prompt)
        lines = response.split('\n')
        is_complete = lines[0].lower().startswith('oui')
        answer = '\n'.join(lines[1:]) if is_complete else ""
        return is_complete, answer

    def _expand_context(self, query: str, relevant_nodes) -> Tuple[str, List[int], Dict[int, str], str]:
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""
        
        priority_queue = []
        distances = {}
        
        print("\nParcours du graphe de connaissances :")
        
        query_concepts = self.generate_concepts(query)
        
        for node, similarity in relevant_nodes:
            concept_similarity = self._calculate_concept_similarity(query_concepts, self.knowledge_graph.graph.nodes[node]['concepts'])
            priority = 1 / (similarity + 0.5 * concept_similarity)  # Ajuster les poids selon vos besoins
            heapq.heappush(priority_queue, (priority, node))
        
        step = 0
        while priority_queue:
            current_priority, current_node = heapq.heappop(priority_queue)
            
            if current_priority > distances.get(current_node, float('inf')):
                continue
            
            if current_node not in traversal_path:
                step += 1
                traversal_path.append(current_node)
                node_content = self.knowledge_graph.graph.nodes[current_node]['content']
                node_concepts = self.knowledge_graph.graph.nodes[current_node]['concepts']
                node_metadata = self.knowledge_graph.graph.nodes[current_node]
                
                filtered_content[current_node] = node_content
                expanded_context += f"\n\nArticle : {node_metadata['head']}\nAuteur : {node_metadata['author']}\nDomaine : {node_metadata['domaine_enccre']}\nContenu : {node_content}"
                
                print(f"\nÉtape {step} - Nœud {current_node}:")
                print(f"Article : {node_metadata['head']}")
                print(f"Auteur : {node_metadata['author']}")
                print(f"Domaine : {node_metadata['domaine_enccre']}")
                print(f"Contenu : {node_content[:100]}...") 
                print(f"Concepts : {', '.join(node_concepts)}")
                print("-" * 50)
                
                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    break
                
                node_concepts_set = set(self.knowledge_graph._lemmatize_concept(c) for c in node_concepts)
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)
                    
                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data['weight']
                        
                        neighbor_concepts = set(self.knowledge_graph.graph.nodes[neighbor]['concepts'])
                        new_concepts = neighbor_concepts - visited_concepts
                        concept_novelty = len(new_concepts) / len(neighbor_concepts)
                        
                        distance = current_priority + (1 / edge_weight) - 0.2 * concept_novelty  # Ajuster le coefficient selon vos besoins
                        
                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))
                
                if final_answer:
                    break

        if not final_answer:
            print("\nGénération de la réponse finale...")
            prompt = f"En vous basant sur le contexte suivant, veuillez répondre à la requête en tenant compte des différents auteurs et domaines mentionnés.\n\nContexte : {expanded_context}\n\nRequête : {query}\n\nRéponse :"
            final_answer = self.generate_response(prompt)

        return expanded_context, traversal_path, filtered_content, final_answer

    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        print(f"\nTraitement de la requête : {query}")
        query_embedding = self.get_embedding(query)
        relevant_nodes = self._retrieve_relevant_nodes(query_embedding)
        expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(query, relevant_nodes)
        
        if not final_answer:
            print("\nGénération de la réponse finale...")
            prompt = f"En vous basant sur le contexte suivant, veuillez répondre à la requête en tenant compte des différents auteurs et domaines mentionnés.\n\nContexte : {expanded_context}\n\nRequête : {query}\n\nRéponse :"
            final_answer = self.generate_response(prompt)
        else:
            print("\nRéponse complète trouvée pendant le parcours.")
        
        return final_answer, traversal_path, filtered_content

    def generate_response(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def _retrieve_relevant_nodes(self, query_embedding, k=5):
        print("\nRecherche des nœuds pertinents...")
        similarities = []
        for node, data in self.knowledge_graph.graph.nodes(data=True):
            node_embedding = data['embedding']
            similarity = self._cosine_similarity(query_embedding, node_embedding)
            similarities.append((node, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))