# this file is used to write global variables
import os, json
_current_dir = os.path.dirname(os.path.abspath(__file__))
backend_port = 5010

data_dir = os.path.join(_current_dir, "../data")
embed_dir = os.path.join(data_dir, "embedding")
result_dir = os.path.join(data_dir, "results")

pred_mapping = {'negative':-1, 'positive':1, 'Positive':1, 'Negative':-1, 'Neutral':0,  'neutral':0}
pred_mapping_reverse = {-1:'negative', 1:'positive', 0:'neutral'}

# interaction_type_mapping = {'R': 'complement_Redundant', 'U1':'conflict_dominant', 'U2':'conflict_dominant', 'S':'distinct'}

video_frame_path = '/data/jianben/vis2024/extracted_frames'