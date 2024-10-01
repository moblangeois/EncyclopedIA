import json
import os
from neo4j import GraphDatabase

class KnowledgeGraphIngestor:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )

    def close(self):
        self.neo4j_driver.close()

    def load_graph_from_jsonld(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            jsonld_data = json.load(f)

        try:
            with self.neo4j_driver.session() as session:
                for item in jsonld_data:
                    if item["@type"] == "Article":
                        # Extraction des informations de l'article
                        article_id = item.get("@id", "")
                        title = item.get("title", "")
                        author = item.get("authors", "Unknown")
                        content = item.get("content", "")
                        url = item.get("url", "URL non disponible")
                        embedding = item.get("embedding", None)

                        # Création du nœud Article
                        session.run(
                            """
                            MERGE (a:Article {id: $id})
                            SET a.title = $title,
                                a.author = $author,
                                a.content = $content,
                                a.url = $url,
                                a.embedding = $embedding
                            """,
                            id=article_id,
                            title=title,
                            author=author,
                            content=content,
                            url=url,
                            embedding=embedding,
                        )

                        # Ajout des relations basées sur les triples
                        for triple in item.get("triples", []):
                            subject = triple.get("subject", "")
                            predicate = triple.get("predicate", "")
                            obj = triple.get("object", "")

                            session.run(
                                """
                                MATCH (a:Article {id: $subject})
                                MERGE (b:Entity {name: $object})
                                MERGE (a)-[r:RELATION {type: $predicate}]->(b)
                                """,
                                subject=subject,
                                predicate=predicate,
                                object=obj,
                            )

            print("Le graphe a été chargé avec succès dans Neo4j.")

        except Exception as e:
            print(f"Erreur lors de l'ingestion des données dans Neo4j : {str(e)}")
            raise e

        finally:
            self.neo4j_driver.close()


if __name__ == "__main__":
    # Spécifiez les détails de connexion à Neo4j
    neo4j_uri = "bolt://localhost:7687"  # URI de Neo4j (port 7687 par défaut)
    neo4j_user = "neo4j"
    neo4j_password = "password123"

    # Chemin du fichier JSON-LD à ingérer
    jsonld_file_path = "EDdA_knowledge_graph.jsonld"  # Assurez-vous que le chemin est correct

    # Créer l'ingestor et charger les données
    ingestor = KnowledgeGraphIngestor(neo4j_uri, neo4j_user, neo4j_password)
    ingestor.load_graph_from_jsonld(jsonld_file_path)
    ingestor.close()
