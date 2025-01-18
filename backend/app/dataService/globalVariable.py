# this file is used to write global variables
import os, json
_current_dir = os.path.dirname(os.path.abspath(__file__))
backend_port = 5010

data_dir = os.path.join(_current_dir, "../data")
embed_dir = os.path.join(data_dir, "embedding")
result_dir = os.path.join(data_dir, "results")
models_dir = os.path.join(_current_dir, "../models")

llama_dir = os.path.join(models_dir, "llama")
llava_dir = os.path.join(models_dir, "llava")

# Model paths
LLAMA_MODEL_PATH = os.path.join(llama_dir, "llama-2-7b-chat.gguf")
LLAVA_MODEL_PATH = os.path.join(llava_dir, "llava-v1.5-13b")

pred_mapping = {'negative':-1, 'positive':1, 'Positive':1, 'Negative':-1, 'Neutral':0, 'neutral':0}
pred_mapping_reverse = {-1:'negative', 1:'positive', 0:'neutral'}

# interaction_type_mapping = {'R': 'complement_Redundant', 'U1':'conflict_dominant', 'U2':'conflict_dominant', 'S':'distinct'}

video_frame_path = '/data/jianben/vis2024/extracted_frames'

LLAVA_CONFIG = {
    'model_path': LLAVA_MODEL_PATH,
    'model_base': 'liuhaotian/llava-v1.5-13b',
    'conv_mode': 'llava_v1',
    'max_new_tokens': 512
}

# Add Llama configuration
LLAMA_CONFIG = {
    'model_path': LLAMA_MODEL_PATH,
    'n_ctx': 4096,
    'n_gpu_layers': -1,
    'verbose': False,
    'max_tokens': 512,
    'top_p': 0.95,
    'temperature': 0.7
}

