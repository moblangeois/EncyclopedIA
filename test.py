import ollama
import chromadb
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import os
import pickle
import logging
import time
from requests.exceptions import Timeout

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="encyclopedia")

    def process_tsv(self, file_path, chunk_size=100):
        logging.info("Début du chargement et traitement du fichier TSV...")
        all_data = []
        total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
        processed_rows = 0

        for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunk_size):
            logging.info(f"Traitement du chunk {processed_rows}-{processed_rows+len(chunk)}")
            chunk['content'] = chunk['content'].astype(str).replace('nan', '')
            chunk = chunk[chunk['content'].str.strip() != '']
            
            chunk_embeddings = []
            
            for _, row in chunk.iterrows():
                try:
                    embedding = ollama.embeddings(model="mxbai-embed-large", prompt=row['content'], timeout=30)['embedding']
                    if embedding and not all(np.isnan(embedding)):
                        chunk_embeddings.append(embedding)
                        self.collection.add(
                            ids=[str(row['id_enccre'])],
                            embeddings=[embedding],
                            documents=[row['content']],
                            metadatas=[{
                                'volume': row['volume'],
                                'numero': row['numero'],
                                'head': row['head'],
                                'author': row['author'],
                                'domaine_enccre': row['domaine_enccre']
                            }]
                        )
                    else:
                        logging.warning(f"Embedding invalide pour l'article {row['id_enccre']}")
                        chunk_embeddings.append(None)
                except Exception as e:
                    logging.error(f"Erreur lors du traitement de l'article {row['id_enccre']}: {e}")
                    chunk_embeddings.append(None)

            chunk['embedding'] = chunk_embeddings
            all_data.append(chunk)
            processed_rows += len(chunk)
            logging.info(f"Progression: {processed_rows}/{total_rows} lignes traitées")

        df = pd.concat(all_data, ignore_index=True)
        logging.info("Traitement du fichier TSV terminé")
        return df

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.edges_threshold = 0.8

    def build_graph(self, df):
        logging.info("Construction du graphe de connaissances...")
        valid_embeddings = []
        valid_node_ids = []
        
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Création des noeuds"):
            node_id = str(row['id_enccre'])
            if isinstance(row['embedding'], list) and not all(np.isnan(row['embedding'])):
                self.graph.add_node(node_id, **row.to_dict())
                valid_embeddings.append(row['embedding'])
                valid_node_ids.append(node_id)
        
        if valid_embeddings:
            self._add_edges(valid_embeddings, valid_node_ids)
        else:
            logging.warning("Aucun embedding valide trouvé. Impossible de créer des arêtes.")

    def _add_edges(self, embeddings, node_ids):
        logging.info("Ajout des arêtes au graphe...")
        similarity_matrix = cosine_similarity(embeddings)
        num_nodes = len(embeddings)
        
        for i in tqdm(range(num_nodes), desc="Création des arêtes"):
            for j in range(i+1, num_nodes):
                similarity_score = similarity_matrix[i][j]
                if similarity_score > self.edges_threshold:
                    self.graph.add_edge(node_ids[i], node_ids[j], weight=similarity_score)

    def get_related_nodes(self, node_id, depth=1):
        related_nodes = set()
        current_nodes = {node_id}
        
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                next_nodes.update(self.graph.neighbors(node))
            related_nodes.update(next_nodes)
            current_nodes = next_nodes
        
        return related_nodes

class QueryEngine:
    def __init__(self, collection, knowledge_graph):
        self.collection = collection
        self.knowledge_graph = knowledge_graph

    def query(self, query: str):
        logging.info("Traitement de la requête...")
        embedding = ollama.embeddings(model="mxbai-embed-large", prompt=query)['embedding']
        results = self.collection.query(query_embeddings=[embedding], n_results=5)
        
        context = ""
        for doc_id, doc_content in zip(results['ids'][0], results['documents'][0]):
            related_nodes = self.knowledge_graph.get_related_nodes(doc_id)
            for related_id in related_nodes:
                related_content = self.knowledge_graph.graph.nodes[related_id]['content']
                context += f"\n{related_content}"
            context += f"\n{doc_content}"
        
        response = ollama.generate(
            model="llama3.1:8b-instruct-q4_0",
            prompt=f"Using this context: {context}\nRespond to this query: {query}"
        )

        return response['response'], results['ids'][0], results['documents'][0]

class GraphRAG:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.knowledge_graph = KnowledgeGraph()
        self.query_engine = None

    def process_tsv(self, file_path):
        if os.path.exists('processed_data.pkl'):
            logging.info("Chargement des données pré-traitées...")
            with open('processed_data.pkl', 'rb') as f:
                df = pickle.load(f)
            self.document_processor.collection = self.document_processor.client.create_collection(name="encyclopedia")
            for _, row in df.iterrows():
                self.document_processor.collection.add(
                    ids=[str(row['id_enccre'])],
                    embeddings=[row['embedding']],
                    documents=[row['content']],
                    metadatas=[{
                        'volume': row['volume'],
                        'numero': row['numero'],
                        'head': row['head'],
                        'author': row['author'],
                        'domaine_enccre': row['domaine_enccre']
                    }]
                )
        else:
            logging.info("Traitement du fichier TSV...")
            df = self.document_processor.process_tsv(file_path)
            with open('processed_data.pkl', 'wb') as f:
                pickle.dump(df, f)

        if os.path.exists('knowledge_graph.pkl'):
            logging.info("Chargement du graphe de connaissances...")
            with open('knowledge_graph.pkl', 'rb') as f:
                self.knowledge_graph = pickle.load(f)
        else:
            logging.info("Construction du graphe de connaissances...")
            self.knowledge_graph.build_graph(df)
            with open('knowledge_graph.pkl', 'wb') as f:
                pickle.dump(self.knowledge_graph, f)

        self.query_engine = QueryEngine(self.document_processor.collection, self.knowledge_graph)

    def query(self, query: str):
        return self.query_engine.query(query)

if __name__ == "__main__":
    graph_rag = GraphRAG()
    graph_rag.process_tsv("data/EDdA_dataframe_withContent_test.tsv")
    
    while True:
        user_query = input("Entrez votre requête (ou 'q' pour quitter): ")
        if user_query.lower() == 'q':
            break
        response, doc_ids, doc_contents = graph_rag.query(user_query)
        print(f"Réponse: {response}")
        print(f"Documents pertinents: {doc_ids}")
        print(f"Contenu des documents: {doc_contents}")