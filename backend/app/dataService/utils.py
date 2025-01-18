# This file is used to write utils functions
import sys
import time
import tqdm
import pickle
import random
import json
import os
# # os.environ["OPENAI_API_KEY"] = ""
import umap
import numpy as np
import pandas as pd
import openai
import concurrent.futures
from sklearn.model_selection import train_test_split
from hdbscan import HDBSCAN
from collections import defaultdict
from sklearn.cluster import KMeans, AgglomerativeClustering
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from openai import OpenAI
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin
import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from timm import create_model
from torchvision import transforms
from PIL import Image
from llama_cpp import Llama
import google.generativeai as genai
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
# from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates 
# from sentence_transformers import SentenceTransformer 

try:
    import globalVariable as GV
except:
    import app.dataService.globalVariable as GV
# try:
#     import globalVariable as GV
# except:
#     import app.dataService.globalVariable as GV

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(
        "OPENAI_API_KEY environment variable must be set when using OpenAI API."
    )
openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI()

def create_batches(lst, n):
    """Yield successive n-sized batches from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# def process_video(videoid = '', model_config = "gpt-3.5-turbo-0125", temperature_config = 0, prompts = {}):
#     try:
#         response_json = client.chat.completions.create(
#             model= model_config,
#             temperature= temperature_config,
#             response_format= {"type": "json_object"},
#             messages= prompts[videoid]
#         )
#         return videoid, response_json.choices[0].message.content, response_json.choices[0].message.role
#     except Exception as e:
#         print(f'Error occurred with {videoid}: {e}, retrying...')
#         time.sleep(10)  # Wait and retry
#         return process_video(videoid, prompts)

def process_video(videoid='', model_config=None, temperature_config=0, prompts={}):
    try:
        # Parse model configuration
        provider = model_config.get('provider', 'openai') if isinstance(model_config, dict) else 'openai'

        if provider == 'openai':
            model_name = model_config if isinstance(model_config, str) else model_config['model_name']
            
            # Check if using GPT-4 Vision
            if 'vision' in model_name:
                # Process messages to include image URLs/base64
                vision_messages = []
                for msg in prompts[videoid]:
                    if msg['role'] == 'user':
                        content = []
                        # Add text if present
                        if isinstance(msg['content'], str):
                            content.append({"type": "text", "text": msg['content']})
                        # Add images if present
                        if 'images' in msg:
                            for image in msg['images']:
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image if isinstance(image, str) else image.get('url'),
                                        # Optionally support base64
                                        # "url": f"data:image/jpeg;base64,{image}" if isinstance(image, str) else image.get('url')
                                    }
                                })
                        vision_messages.append({"role": msg['role'], "content": content})
                    else:
                        vision_messages.append(msg)
                response_json = client.chat.completions.create(
                    model=model_name,
                    temperature=temperature_config,
                    response_format={"type": "json_object"},
                    messages=vision_messages,
                    max_tokens=model_config.get('max_tokens', 4096)
                )
            else:
                # Standard OpenAI text completion
                response_json = client.chat.completions.create(
                    model=model_name,
                    temperature=temperature_config,
                    response_format={"type": "json_object"},
                    messages=prompts[videoid]
                )
            return videoid, response_json.choices[0].message.content, response_json.choices[0].message.role
                

        elif provider == 'llava':
            model_path = model_config['model_path']
            model_name = get_model_name_from_path(model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=model_config.get('model_base', None),
                model_name=model_name
            )
            images = prompts[videoid].get('images', [])
            if images:
                image_tensors = [image_processor(image) for image in images]
            else:
                image_tensors = []

            # Prepare conversation
            conv = conv_templates[model_config.get('conv_mode', 'llava_v1')].copy()
            conv.append_message(conv.roles[0], prompts[videoid][-1]['content'])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Generate response
            input_ids = tokenizer([prompt]).input_ids
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=torch.tensor(input_ids).cuda(),
                    images=torch.stack(image_tensors).cuda() if image_tensors else None,
                    temperature=temperature_config,
                    max_new_tokens=model_config.get('max_new_tokens', 512),
                    do_sample=True
                )
            
            response = tokenizer.decode(output_ids[0, input_ids[0].shape[0]:], skip_special_tokens=True)
            return videoid, response, 'assistant'
        
        elif provider == 'gemini':
            genai.configure(api_key=model_config['api_key'])
            model = genai.GenerativeModel('gemini-pro-vision')
            images = prompts[videoid].get('images', [])
            response = model.generate_content(
                [prompts[videoid][-1]['content'], *images],
                generation_config={"temperature": temperature_config}
            )
            return videoid, response.text, 'assistant'

        elif provider == 'llama':
            # Initialize Llama model if not already loaded
            if not hasattr(process_video, 'llama_model'):
                process_video.llama_model = Llama(
                    model_path=model_config['model_path'],
                    n_ctx=model_config.get('n_ctx', 4096),
                    n_gpu_layers=model_config.get('n_gpu_layers', -1),
                    verbose=model_config.get('verbose', False)
                )
            messages = prompts[videoid]
            formatted_prompt = ""
            for msg in messages:
                if msg['role'] == 'system':
                    formatted_prompt += f"System: {msg['content']}\n\n"
                elif msg['role'] == 'user':
                    formatted_prompt += f"User: {msg['content']}\n\n"
                elif msg['role'] == 'assistant':
                    formatted_prompt += f"Assistant: {msg['content']}\n\n"
            formatted_prompt += "Assistant: "

            # Generate response
            response = process_video.llama_model.create_completion(
                prompt=formatted_prompt,
                max_tokens=model_config.get('max_tokens', 512),
                temperature=temperature_config,
                top_p=model_config.get('top_p', 0.95),
                stop=["User:", "\n\nUser:", "System:"],
                echo=False
            )
            return videoid, response['choices'][0]['text'].strip(), 'assistant'
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    except Exception as e:
        print(f'Error occurred with {videoid}: {e}, retrying...')
        time.sleep(10)
        return process_video(videoid, model_config, temperature_config, prompts)
    


def model_batch_generation(BATCH_SIZE = 10, video_list = [], model_config = "gpt-3.5-turbo-0125", temperature_config = 0, prompts = {}):
    response_dict = {}
    print('video_list', video_list)
    time_stamp_str = ''
    for t in time.localtime()[0: 6]:
        if(len(str(t)) <= 2):
            t = '%.2d' %t
        time_stamp_str += str(t)
    model_run_result_path = GV.result_dir + '_t' + time_stamp_str
    os.makedirs(model_run_result_path, exist_ok=True)

    for batch in create_batches(video_list, BATCH_SIZE):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_videoid = {executor.submit(process_video, videoid, model_config, temperature_config, prompts): videoid for videoid in batch}
            for future in concurrent.futures.as_completed(future_to_videoid):
                videoid = future_to_videoid[future]
                try:
                    data = future.result()
                    response_dict[videoid] = {'content': data[1], 'role': data[2]}
                    print(videoid)
                except Exception as exc:
                    print(f'{videoid} generated an exception: {exc}')

                with open(os.path.join(model_run_result_path, 'result.json'), 'w') as file:
                    json.dump(response_dict, file)

    return response_dict


def gpt4_generation(prompt):
    try:
        response_json = client.chat.completions.create(
            model= "gpt-4-1106-preview",
            temperature= 0,
            # response_format= {"type": "json_object"},
            messages= prompt
        )
        return response_json.choices[0].message.content
    except Exception as e:
        print(f'Error occurred : {e}, retrying...')
        time.sleep(10)  # Wait and retry
        return gpt4_generation(prompt)


def low_level_principle_generation(videoid = '', model_config = "gpt-4-1106-preview", temperature_config = 0, prompts = {}):
    try:
        response_json = client.chat.completions.create(
            model= model_config,
            temperature= temperature_config,
            response_format= {"type": "json_object"},
            messages= prompts[videoid]
        )
        return videoid, response_json.choices[0].message.content
    except Exception as e:
        print(f'Error occurred with {videoid}: {e}, retrying...')
        time.sleep(10)  # Wait and retry
        return gpt4_generation(videoid, prompts)

    
def low_principle_batch_generation(BATCH_SIZE = 10, video_list = [], model_config = "gpt-4-1106-preview", temperature_config = 0, prompts = {}):
    response_dict = {}
    for batch in create_batches(video_list, BATCH_SIZE):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_videoid = {executor.submit(low_level_principle_generation, videoid, model_config, temperature_config, prompts): videoid for videoid in batch}
            for future in concurrent.futures.as_completed(future_to_videoid):
                videoid = future_to_videoid[future]
                try:
                    data = future.result()
                    response_dict[videoid] = data[1]
                    print(videoid)
                except Exception as exc:
                    print(f'{videoid} generated an exception: {exc}')

                # with open(os.path.join(model_run_result_path, 'result.json'), 'w') as file:
                #     json.dump(response_dict, file)

    return response_dict



### gpt-embedding model
def get_text_embedding_gpt(text, model = 'text-embedding-3-small'):
    response = client.embeddings.create(
        model= model,
        input= text,
        dimensions = 512
        )
    # Extracting the embedding
    embedding = response.data[0].embedding
    return embedding



def cosine_similarity(v1, v2):
    """
    Input:
        - v1: N*M matrix; N: data points, M: feature dimension (np.array)
        - v2: N*M matrix; N: data points, M: feature dimension (np.array)
    Output:
        - cosine similarity: N*1 matrix
    """
    # print("type of v1: ", type(v1))
    # print("type of v2: ", type(v2))
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2.T) / (np.linalg.norm(v1, axis=1, keepdims=True) * np.linalg.norm(v2, axis=1, keepdims=True))
    

### sentence-transformer model
# def get_text_embedding_sentence_transformer(text):
#     model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#     embedding = model.encode(text, convert_to_tensor=True)
#     return embedding


def compute_heuristic(y1, y2, y):
    R = -np.abs(y1-y)-np.abs(y1-y2)-np.abs(y2-y)
    U1 = np.abs(y2-y)+np.abs(y1-y2)-np.abs(y1-y)
    U2 = np.abs(y1-y)+np.abs(y1-y2)-np.abs(y2-y)
    S = np.abs(y1-y)+np.abs(y2-y)
    return R, U1, U2, S


def relation_computation(R, U1, U2, S):
    if (R == 0 and U1 == 0 and U2 == 0 and S == 0):
        return "R"
    elif (R == -2 and U1 == 2 and U2 == 0 and S == 1):
        return "U1"
    elif (R == -2 and U1 == 0 and U2 == 2 and S == 1):
        return "U2"
    elif (R == -4 and U1 == 2 and U2 == 2 and S == 2):
        return 'S'
    elif (R == -4 and U1 == 2 and U2 == 0 and S == 3):
        return 'S'
    elif (R == -4 and U1 == 0 and U2 == 2 and S == 3):
        return 'S'
    elif (R == -2 and U1 == 0 and U2 == 0 and S == 2):
        return 'S'            
    elif (R == -4 and U1 == 4 and U2 == 0 and S == 2):
        return 'U1' 
    elif (R == -4 and U1 == 0 and U2 == 4 and S == 2):
        return 'U2'             
    else:
        return "undecided"

def modality_type_computation(y1, y2):
    if y1 == y2:
        return 'complement'
    else:
        return 'conflict'

def reduce_project_features_umap(feats, labels = None, n_neighbors = 15, metric='cosine', random_state: int=None):
    """
    Input:
        - feats: N*M matrix; N: data points, M: feature dimension
        - labels: list, target labels/classes
    Output:
        - 2D positions: N*2 matrix
    Hyperparameters:
        default: n_neighbor: 15; metric: cosine similarity
        - https://umap-learn.readthedocs.io/en/latest/parameters.html
    """
    if labels is not None:
        embedding = umap.UMAP(metric=metric, n_neighbors=n_neighbors, random_state=random_state).fit_transform(feats, y=labels)
    else:
        embedding = umap.UMAP(metric=metric, n_neighbors=n_neighbors, random_state=random_state).fit_transform(feats, y=labels)
    return embedding


def convert_label(label):
    if (label > -0.5 and label < 0.5):
        gt = 0
    elif (label >= 0.5 and label <= 3):
        gt = 1
    elif (label >= -3 and label <= -0.5):
        gt = -1
    return gt

def dict_to_labels(data_dict):
    samples = list(data_dict.keys())
    labels = np.array([convert_label(value['label']) for value in data_dict.values()])
    return samples, labels


def split_dataset_stratified(X, y, test_size = 0.2, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)
    return X_train, X_test, y_train, y_test


def compute_video_frame(data_dict):
    data_path = GV.video_frame_path
    for key in data_dict.keys():
        file_path = os.path.join(data_path, key)
        file_num = 0
        if os.path.exists(file_path):
            # print(os.listdir(file_path))
            for entry in os.listdir(file_path):
                if entry.endswith(".jpg"):
                    file_num += 1
        else:
            print("no such file")
        print(file_num)
        data_dict[key]['frame_num'] = file_num
    return data_dict




class Cue():
    def __init__(self, cue_type: str, cue_text, sentiment, embedding):
        self.cue_type = cue_type
        self.cue_text = cue_text
        self.sentiment = sentiment
        self.embedding = embedding

    def __eq__(self, other):
        return self.cue_type == other.cue_type and self.cue_text == other.cue_text

class DataUnit():
    def __init__(self, key, label, description, prediction, visual_cues, language_cues):
        self.key = key
        self.label = label
        self.description = description
        self.prediction = prediction
        self.visual_cues = visual_cues
        self.language_cues = language_cues
        self.cues = [*self.visual_cues, *self.language_cues]


class ClusteredDataUnit:
    def __init__(self, key, label, description, prediction, visual_cue_clusters, language_cue_clusters, visual_cues, language_cues, cues):
        self.key = key
        self.label = label
        self.description = description
        self.prediction = prediction
        self.visual_cue_clusters = visual_cue_clusters
        self.language_cue_clusters = language_cue_clusters
        self.visual_cues = visual_cues
        self.language_cues = language_cues
        self.cues = cues



def parse_reasoning(data_result_dict, embedding_dict, video_idx):
    visual_cues = data_result_dict[video_idx]['reasoning']['visual_cues']
    language_cues = data_result_dict[video_idx]['reasoning']['language_cues']
    visual_cue_objs = []
    language_cue_objs = []
    for cue in visual_cues:
        embedding = embedding_dict[video_idx][cue[0]]
        cue_obj = Cue("visual", cue[0], cue[1], embedding) if len(cue) == 2 else Cue("visual", cue[0], cue[0], embedding)
        visual_cue_objs.append(cue_obj)
    for cue in language_cues:
        embedding = embedding_dict[video_idx][cue[0]]
        cue_obj = Cue("language", cue[0], cue[1], embedding) if len(cue) == 2 else Cue("language", cue[0], cue[0], embedding)
        language_cue_objs.append(cue_obj)
    return DataUnit(video_idx, data_result_dict[video_idx]['label'], data_result_dict[video_idx]['description'], data_result_dict[video_idx]['prediction'], visual_cue_objs, language_cue_objs)
    

class CueCluster:
    def __init__(self, cue_type, cluster_id, cues):
        self.cue_type = cue_type
        self.label = cluster_id
        self.cluster_id = cue_type + ":" + str(cluster_id)
        self.cues = cues  # List of cues in this cluster
        self.centroid = np.mean([cue.embedding for cue in cues], axis=0) if cues else None

    def representative_cue(self):
        # Assuming that a smaller distance indicates higher similarity
        return min(self.cues, key=lambda cue: np.linalg.norm(cue.embedding - self.centroid))

    @staticmethod
    def serialize_cue(cue):
        # This method should be adjusted based on the actual structure of your cue object.
        # For demonstration, I'm assuming each cue has 'cue_text' and 'embedding' attributes.
        return {
            'cue_text': cue.cue_text,

            # 'embedding': cue.embedding  # Convert numpy array to list
        }
        
    def to_dict(self):
        # Serialize the CueCluster object into a dictionary
        return {
            'cue_type': self.cue_type,
            'cluster_id': self.cluster_id,
            # 'centroid': self.centroid.tolist(),  # Convert numpy array to list
            'representative_cue': self.serialize_cue(self.representative_cue()),
            'cues': [self.serialize_cue(cue) for cue in self.cues]
        }


def cluster_cues(cues, cue_type, method='kmeans', min_cluster_size=5, min_samples=10, n_clusters=5):
    # Extract embeddings and apply HDBSCAN
    embeddings = np.array([cue.embedding for cue in cues])
    if method == 'hdbscan':
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(embeddings)
    elif method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(embeddings)
    elif method == 'agglomerative':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(embeddings)
    else:
        raise ValueError("Unsupported clustering method. Choose from 'hdbscan', 'kmeans', or 'agglomerative'.")

    # Group cues into clusters
    cluster_dict = defaultdict(list)
    for cue, label in zip(cues, labels):
        cluster_dict[label].append(cue)

    # Create CueCluster objects
    cue_clusters = {label: CueCluster(cue_type, label, cluster_cues) for label, cluster_cues in cluster_dict.items()}

    return cue_clusters


def process_and_cluster_cues(data_units):
    def generate_cue_key(cue):
        # Create a unique identifier for the cue based on its content
        # # Adjust the content based on the actual attributes of your Cue objects
        # return cue.cue_text + '_' + '_'.join(map(str, cue.embedding))
        return cue.cue_text

    # Dictionaries to hold unique cues, keyed by their generated unique identifier
    unique_visual_cues = {}
    unique_language_cues = {}

    visual_cue_num = 0
    language_cue_num = 0
    # Extract unique cues from each data unit
    for data_unit in data_units:
        for cue in data_unit.visual_cues:
            visual_cue_num += 1
            key = generate_cue_key(cue)
            if key not in unique_visual_cues:
                unique_visual_cues[key] = cue


        for cue in data_unit.language_cues:
            language_cue_num += 1
            key = generate_cue_key(cue)
            if key not in unique_language_cues:
                unique_language_cues[key] = cue


    # Convert the unique cues dictionaries back to lists
    all_unique_visual_cues = list(unique_visual_cues.values())
    all_unique_language_cues = list(unique_language_cues.values())
    print('all_unique_visual_cues',len(all_unique_visual_cues))
    print('all_visual_cues', visual_cue_num) 
    print('all_unique_language_cues',len(all_unique_language_cues))
    print('all_language_cues', language_cue_num) 

    # Cluster the unique cues
    visual_cue_clusters = cluster_cues(all_unique_visual_cues, "visual")
    language_cue_clusters = cluster_cues(all_unique_language_cues, "language")

    return visual_cue_clusters, language_cue_clusters

def create_clustered_data_units(data_units, visual_cue_clusters, language_cue_clusters):
    clustered_data_units = []
    unassigned_cluster_id_visual = max(visual_cue_clusters.keys(), default=0) + 1
    unassigned_cluster_id_language = max(language_cue_clusters.keys(), default=0) + 1

    for data_unit in data_units:
        updated_visual_cues = []
        for cue in data_unit.visual_cues:
            cluster = next((visual_cue_clusters[c] for c in visual_cue_clusters if cue in visual_cue_clusters[c].cues), None)
            if cluster is None:
                # Create a new cluster for the unassigned cue
                cluster = CueCluster("visual", unassigned_cluster_id_visual, [cue])
                visual_cue_clusters[unassigned_cluster_id_visual] = cluster
                unassigned_cluster_id_visual += 1
            updated_visual_cues.append(cluster)

        updated_language_cues = []
        for cue in data_unit.language_cues:
            cluster = next((language_cue_clusters[c] for c in language_cue_clusters if cue in language_cue_clusters[c].cues), None)
            if cluster is None:
                # Create a new cluster for the unassigned cue
                cluster = CueCluster("language", unassigned_cluster_id_language, [cue])
                language_cue_clusters[unassigned_cluster_id_language] = cluster
                unassigned_cluster_id_language += 1
            updated_language_cues.append(cluster)

        clustered_data_unit = ClusteredDataUnit(
            key=data_unit.key,
            label=data_unit.label,
            description=data_unit.description,
            prediction=data_unit.prediction,
            visual_cue_clusters= updated_visual_cues,
            language_cue_clusters= updated_language_cues,
            visual_cues = data_unit.visual_cues,
            language_cues = data_unit.language_cues,
            cues = data_unit.cues
        )

        clustered_data_units.append(clustered_data_unit)
        joint_clusters = merge_visual_language_clusters(visual_cue_clusters, language_cue_clusters, updated_visual_cues, updated_language_cues)

    print('unsigned_cluster_id_visual', unassigned_cluster_id_visual)
    print('unsigned_cluster_id_language', unassigned_cluster_id_language)
    return clustered_data_units, joint_clusters

def merge_visual_language_clusters(visual_cue_dict, language_cue_dict, updated_visual_cue_clusters, updated_language_cue_clusters):
    joint_dict = {}
    for cluster in visual_cue_dict.values():
        joint_dict[cluster.cluster_id] = cluster
    for cluster in language_cue_dict.values():
        joint_dict[cluster.cluster_id] = cluster
    ### why do the following steps?
    for cluster in updated_visual_cue_clusters:
        joint_dict[cluster.cluster_id] = cluster
    for cluster in updated_language_cue_clusters:
        joint_dict[cluster.cluster_id] = cluster
    return joint_dict


def prepare_transactions(clustered_data_units):
    transactions = []

    for unit in clustered_data_units:
        # Combine visual and language cue clusters, extracting their cluster IDs
        transaction = set(cluster.cluster_id for cluster in unit.visual_cue_clusters + unit.language_cue_clusters)
        transactions.append(list(transaction))

    return transactions



def run_pattern_mining(transactions, min_support=0.05):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    print(frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    return rules, frequent_itemsets



def combined_data(frequent_itemsets, rules, cue_clusters):
    # Convert CueCluster objects to a serializable format
    serialized_clusters = {k: cluster.to_dict() for k, cluster in cue_clusters.items()}
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: sorted(list(x)))

    combined_data = {
        'frequent_itemsets': frequent_itemsets.to_dict(orient='records'),
        # 'rules': rules.to_dict(orient='records'),
        'clusters': serialized_clusters
    }
    return combined_data

def find_frequent_set_instances(clustered_data_units, combined_data):
    video_dict = {}
    for idx, item in enumerate(combined_data['frequent_itemsets']):
        combined_data['frequent_itemsets'][idx]['video_instances'] = []

    for unit in clustered_data_units:
        transaction = set(cluster.cluster_id for cluster in unit.visual_cue_clusters + unit.language_cue_clusters)
        for idx, item in enumerate(combined_data['frequent_itemsets']):
            itemsets = item['itemsets']
            pattern_set = set(pattern for pattern in itemsets)
            if set(itemsets).issubset(transaction):
                combined_data['frequent_itemsets'][idx]['video_instances'].append(unit.key)
    
        # video_dict[unit.key] = {}
        # video_dict[unit.key]['label'] = unit.label
        # video_dict[unit.key]['prediction'] = unit.prediction
        # video_dict[unit.key]['cue_list']  = {}
        # cluster_list = [cluster.cluster_id for cluster in unit.visual_cue_clusters + unit.language_cue_clusters]
        # for idx, cue in enumerate(unit.cues):
        #     cue_type = cue.cue_type
        #     cue_text = cue.cue_text
        #     cue_sentiment = cue.sentiment
        #     cue_embedding = cue.embedding
        #     cue_cluster = cluster_list[idx]
        #     video_dict[unit.key]['cue_list'][cue_cluster] = {'cue_type': cue_type, 'cue_text': cue_text, 'cue_sentiment': cue_sentiment, 'cue_embedding': cue_embedding}
    
    return combined_data


def split_items(input_string):
    # Split the string into items based on the pattern of a newline followed by a digit and a period
    items = re.split(r'\n\n\d+\.', input_string)
    # The first item might start with a digit and period (because it won't be split by the regex), so remove it if present
    items[0] = re.sub(r'^\d+\.', '', items[0]).strip()
    # Clean and strip whitespace for all items
    cleaned_items = [item.strip() for item in items]
    return cleaned_items


def reduce_project_features_tsne(feats, labels=None, perplexity=30, metric='cosine', random_state=None):
    """
    Input:
        - feats: N*M matrix; N: data points, M: feature dimension
        - labels: list, target labels/classes (not used in t-SNE directly)
    Output:
        - 2D positions: N*2 matrix
    Hyperparameters:
        default: perplexity: 30; metric: cosine similarity
        - https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """
    # Convert feats to a NumPy array if it is not already
    if not isinstance(feats, np.ndarray):
        feats = np.array(feats)

    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        random_state=random_state,
        init='pca',           # Explicitly set initialization to 'pca' to match future default
        learning_rate='auto', # Explicitly set learning rate to 'auto' to match future default
        square_distances=True # Ensures consistent distance calculation behavior
    ).fit_transform(feats)
    
    return embedding


class MultimodalModel(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', vit_model_name='vit_base_patch16_224'):
        super(MultimodalModel, self).__init__()
        
        # Visual Encoder (ViT)
        self.vit = create_model(vit_model_name, pretrained=True)
        self.vit.head = nn.Identity()  # Remove classification head for feature extraction
        
        # Text Encoder (BERT)
        self.tokenizer = BertTokenizer.from_pretrained(text_model_name)
        self.bert = BertModel.from_pretrained(text_model_name)
        
        # Attention Layer to fuse video and text embeddings
        self.attention_layer = nn.MultiheadAttention(embed_dim=768, num_heads=8)  # Use appropriate dim
        
        # Output layer to generate the final embedding
        self.output_layer = nn.Linear(768, 256)  # Example output dim

    def forward(self, video_frames, text_input):
        # Process video frames through ViT
        vit_features = self.process_video(video_frames)
        
        # Process text input through BERT
        text_features = self.process_text(text_input)
        
        # Apply attention layer to combine the video and text features
        combined_features = torch.cat([vit_features.unsqueeze(0), text_features.unsqueeze(0)], dim=0)
        attn_output, _ = self.attention_layer(combined_features, combined_features, combined_features)
        
        # Use the attention output as the embedding representation
        embedding_output = self.output_layer(attn_output.mean(dim=0))  # Pool the attention output
        
        return embedding_output

    def process_video(self, video_frames):
        # Assuming video_frames is a list of PIL images or torch tensors
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        frames_tensor = torch.stack([transform(frame) for frame in video_frames])  # Stack frames to create a tensor
        frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension
        
        # Extract features using ViT
        vit_features = self.vit(frames_tensor)
        return vit_features.mean(dim=1)  # Pooling frames

    def process_text(self, text_input):
        # Tokenize and process the text input
        inputs = self.tokenizer(text_input, return_tensors='pt', truncation=True, padding=True)
        text_output = self.bert(**inputs)
        
        # Return the hidden state of the [CLS] token
        return text_output.last_hidden_state.mean(dim=1)

# Step 2: Usage example
def generate_embedding(video_frames, text_input):
    model = MultimodalModel()
    model.eval()
    
    # Forward pass to get the embedding
    with torch.no_grad():
        embedding = model(video_frames, text_input)
    
    return embedding

def cluster_video_instances(embeddings, num_clusters):
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    
    # Get the cluster labels for each embedding
    cluster_labels = kmeans.labels_
    
    return cluster_labels

# def reduce_project_features_umap(feats, labels = None, n_neighbors = 15, metric='cosine', random_state: int=None):
#     """
#     Input:
#         - feats: N*M matrix; N: data points, M: feature dimension
#         - labels: list, target labels/classes
#     Output:
#         - 2D positions: N*2 matrix
#     Hyperparameters:
#         default: n_neighbor: 15; metric: cosine similarity
#         - https://umap-learn.readthedocs.io/en/latest/parameters.html
#     """
#     if labels is not None:
#         embedding = umap.UMAP(metric=metric, n_neighbors=n_neighbors, random_state=random_state).fit_transform(feats, y=labels)
#     else:
#         embedding = umap.UMAP(metric=metric, n_neighbors=n_neighbors, random_state=random_state).fit_transform(feats, y=labels)
#     return embedding


# def extract_and_plot_antecedents(rules, cue_clusters, top_n=10):
#     # Convert antecedents from frozensets to a list of strings (representing cluster IDs)
#     print(rules['antecedents'])
#     filtered_rules = rules[rules['antecedents'].apply(lambda x: len(x) > 1)]
    

#     # Convert antecedents from frozensets to a list of strings (representing cluster IDs)
#     filtered_rules['antecedent_str'] = filtered_rules['antecedents'].apply(lambda x: ', '.join(str(item) for item in x))

#     # Count the occurrences of each antecedent set
#     antecedent_counts = filtered_rules['antecedent_str'].value_counts().head(top_n)
#     print(antecedent_counts)
#     # Plot the top antecedents
#     # antecedent_counts.plot(kind='bar', figsize=(12, 6), color='skyblue')
#     plt.title(f'Top {top_n} Antecedents')
#     plt.ylabel('Occurrences')
#     plt.xlabel('Antecedents')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()
#     # For each top antecedent, display the included cues and their representative cue
#     for antecedent in antecedent_counts.index:
#         cluster_ids = [id_ for id_ in antecedent.split(', ')]
#         for cluster_id in cluster_ids:
#             cluster = cue_clusters[cluster_id]
#             representative_cue = cluster.representative_cue()
#             print(f"Cluster {cluster_id}: Representative Cue - {representative_cue.cue_text}")

# Usage
# Assuming 'mined_patterns' is your DataFrame containing the association rules
# extract_and_plot_antecedents(mined_patterns, top_n=10)


# Assuming 'mined_patterns' is your DataFrame containing the rules
# and 'cue_clusters' is a dictionary mapping cluster IDs to CueCluster objects
            
# cue_clusters = []
# extract_and_plot_antecedents(mined_patterns, joint_clusters, top_n=10)



# def perturbation_based_explanation(example_list):
#     ### the example_list format may be: input, output, reasoning
#     prompt_template = """
#         context: {context}
#         dataset: {example_list}
#         question: "Based on the model's predictions and the given dataset, what appears to be the top five most important features in determining the model’s prediction?"
#         instructions: Think about the question. After explaining your reasoning, provide your answer as the top five most important features ranked from most important to least important, in descending order. Only provide the feature names on the last line. Do not provide any further details on the last line.
#     """

# def prediction_based_explanation(example_list):
#     prompt_template = """
#         context: {context}
#         dataset: {example_list}
#         question: "Based on the model’s predictions and the given dataset, estimate the output for the final input. What appears to be the top five most important features in determining the model’s prediction?"
#         instructions: "Think about the question. After explaining your reasoning, provide your answer as the top five most important features ranked from most important to least important, in descending order. Only provide the feature names on the last line. Do not provide any further details on the last line."
#     """

# def instruction_based_explanation(example_list):
#     prompt_template = """
#         context: {context}
#         dataset: {example_list}
#         question: "Based on the model’s predictions and the given dataset, estimate the output for the final input. What appears to be the top five most important features in determining the model’s prediction?"
#         instructions: "Think about the question. After explaining your reasoning, provide your answer as the top five most important features ranked from most important to least important, in descending order. Only provide the feature names on the last line. Do not provide any further details on the last line."
#     """