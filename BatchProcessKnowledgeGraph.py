import os
import requests
import re
import pandas as pd
import json
import time
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

class SimpleKnowledgeGraph:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        self.df = None

    def load_data(self, file_path):
        self.df = pd.read_csv(file_path, sep='\t')
        self.df['content'] = self.df['content'].astype(str).replace('nan', '')
        self.df = self.df[self.df['content'].str.strip() != '']
        self.df['references'] = self.df['content'].apply(self.extract_references)

    def extract_references(self, text):
        pattern = r'Voyez\s+([^.,;]+)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [match.strip() for match in matches]

    def create_embeddings(self):
        # Créer les fichiers batch
        batch_size = 800
        batch_file = self.df.copy()
        batch_file_name = 'embedding_batch'
        num_files = len(batch_file) // batch_size + 1

        check_and_create_folder('./batch_files')

        batch_input_files = []
        for num_file in range(num_files):
            output_file = f'./batch_files/{batch_file_name}_part{num_file}.jsonl'
            if os.path.exists(output_file):
                os.remove(output_file)
            
            with open(output_file, 'a') as file:
                for index, row in batch_file.iloc[batch_size*num_file : min(batch_size*(num_file+1), len(batch_file))].iterrows():
                    payload = {
                        "custom_id": f"custom_id_{index}",
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {
                            "input": row["content"],
                            "model": "text-embedding-3-small",
                            "encoding_format": "float",
                        }
                    }
                    file.write(json.dumps(payload) + '\n')
            
            batch_input_files.append(self.client.files.create(
                file=open(output_file, "rb"),
                purpose="batch"
            ))

        # Créer les jobs batch
        batch_file_ids = [batch_file.id for batch_file in batch_input_files]
        job_creations = []
        for i, file_id in enumerate(batch_file_ids):
            job_creations.append(self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/embeddings",
                completion_window="24h",
                metadata={
                    "description": f"part_{i}_embeddings"
                }
            ))

        # Monitorer les jobs
        job_ids = [job.id for job in job_creations]
        fail_flag = False
        finished = set()
        while True:
            for job_id in job_ids:
                job = self.client.batches.retrieve(job_id)
                if job.status == "failed":
                    print(f"Job {job_id} has failed with error {job.errors}")
                    fail_flag = True
                    break
                elif job.status == 'in_progress':
                    print(f'Job {job_id} is in progress, {job.request_counts.completed}/{job.request_counts.total} requests completed')
                elif job.status == 'finalizing':
                    print(f'Job {job_id} is finalizing, waiting for the output file id')
                elif job.status == "completed":
                    print(f"Job {job_id} has finished")
                    finished.add(job_id)
                else:
                    print(f'Job {job_id} is in status {job.status}')
            
            if fail_flag or len(finished) == len(job_ids):
                break
            time.sleep(60)

        # Télécharger et traiter les fichiers de sortie
        output_files_ids = [self.client.batches.retrieve(job_id).output_file_id for job_id in job_ids]
        embedding_results = []
        for output_file_id in output_files_ids:
            output_file = self.client.files.content(output_file_id).text
            for line in output_file.split('\n')[:-1]:
                data = json.loads(line)
                custom_id = data.get('custom_id')
                embedding = data['response']['body']['data'][0]['embedding']
                embedding_results.append([custom_id, embedding])

        embedding_df = pd.DataFrame(embedding_results, columns=['custom_id', 'embedding'])
        embedding_df['id'] = embedding_df['custom_id'].apply(lambda x: int(x.split('custom_id_')[1]))
        
        self.df = self.df.reset_index().rename(columns={'index': 'id'})
        self.df = self.df.merge(embedding_df[['id', 'embedding']], on='id', how='left')

    def export_to_jsonld(self, file_path):
        jsonld_data = []
        for _, row in self.df.iterrows():
            article_url = f"http://enccre.academie-sciences.fr/encyclopedie/article/{row['id_enccre']}/"

            node_data = {
                "@context": "http://schema.org",
                "@type": "Article",
                "@id": row['id_enccre'],
                "url": article_url,
                "title": row.get('head', ''),
                "authors": row.get('author', 'Unknown'),
                "content": row['content'],
                "references": row['references'],
                "embedding": row['embedding'].tolist() if isinstance(row['embedding'], np.ndarray) else row['embedding']
            }

            triples = [
                {
                    "subject": node_data["@id"],
                    "predicate": "is_written_by",
                    "object": row.get('author', 'Unknown')
                },
                {
                    "subject": node_data["@id"],
                    "predicate": "belongs_to_domain",
                    "object": row.get('domaine_enccre', 'Unknown')
                }
            ]
            
            for ref in row['references']:
                triples.append({
                    "subject": node_data["@id"],
                    "predicate": "references",
                    "object": ref
                })
            
            node_data["triples"] = triples
            jsonld_data.append(node_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(jsonld_data, f, ensure_ascii=False, indent=2)

        print(f"Graphe exporté au format JSON-LD dans {file_path}")

def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    kg = SimpleKnowledgeGraph(openai_api_key)
    kg.load_data("data/EDdA_dataframe_sample.tsv")
    kg.create_embeddings()
    kg.export_to_jsonld("data/EDdA_knowledge_graph_sample.jsonld")