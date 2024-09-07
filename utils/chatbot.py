import os
import logging
from utils.knowledgegraph import KnowledgeGraph
from utils.queryengine import QueryEngine
from utils.visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

class Chatbot:
    def __init__(self, data_file, openai_api_key):
        self.data_file = data_file
        self.knowledge_graph = None
        self.openai_api_key = openai_api_key
        self.query_engine = None

    def initialize(self):
        graph_file = "knowledge_graph.pkl"
        if os.path.exists(graph_file):
            logging.info("Chargement d'un graphe existant...")
            self.knowledge_graph = KnowledgeGraph.load_graph(graph_file, self.openai_api_key)
        else:
            logging.info("Création d'un nouveau graphe...")
            self.knowledge_graph = KnowledgeGraph(openai_api_key=self.openai_api_key)
            self.knowledge_graph.load_data(self.data_file)
            self.knowledge_graph.create_embeddings_and_concepts()
            self.knowledge_graph.build_graph()
            self.knowledge_graph.save_graph(graph_file)
        
        self.query_engine = QueryEngine(self.knowledge_graph, self.openai_api_key)
        logging.info("Initialisation terminée.")

    def chat(self, user_query):
        try:
            response, traversal_path, filtered_content = self.query_engine.query(user_query)
            
            print(f"Réponse : {response}\n\n")

            # Visualiser le parcours du graphe
            Visualizer.visualize_traversal(self.knowledge_graph.graph, traversal_path)
            
        except Exception as e:
            print(f"Erreur lors du traitement de la requête : {str(e)}")

    def export_graph_jsonld(self, file_path):
        if self.knowledge_graph:
            self.knowledge_graph.export_to_jsonld(file_path)
        else:
            logging.error("Le graphe de connaissances n'a pas été initialisé.")

    def export_graph_rdf(self, file_path):
        if self.knowledge_graph:
            self.knowledge_graph.export_to_rdf(file_path)
        else:
            logging.error("Le graphe de connaissances n'a pas été initialisé.")