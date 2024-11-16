from crypt import methods
import json
import os
import logging
import numpy as np
import pandas as pd
import pickle
from flask import Blueprint, current_app, request, jsonify, send_from_directory, url_for, Response, stream_with_context

# current_app is a proxy to the current application (which is initialized in app.py)
# default backend url: http://localhost:{portnumber}/api

LOG = logging.getLogger(__name__)
api = Blueprint('api', __name__)

@api.route('/')
def index():
    print('main url!')
    return json.dumps('/')

@api.route('/initialization/<datasetType>/<modelType>')
def initialization(datasetType, modelType):
    valid_data_list, k_shot_data_list, extra_test_data_list = current_app.dataService.load_all_data(datasetType, modelType)
    return jsonify({'valid_data_list': valid_data_list, 'k_shot_data_list': k_shot_data_list, 'extra_test_data_list': extra_test_data_list})

@api.route('/load_prompt_template', methods = ['GET'])
def load_prompt_template():
    prompt_template = current_app.dataService.load_prompt_template()
    return jsonify({'prompt_template': prompt_template})

# @api.route('/load_data_for_center_view_table', methods = ['GET'])
# def load_data_for_center_view_table():
#     result = current_app.dataService.generate_results_for_instances(original = True)
#     reasoning_cue_embedding = current_app.dataService.reasoning_cue_embedding
#     frequent_set_data = current_app.dataService.reasoning_pattern_mining(result, reasoning_cue_embedding, list(result.keys()))
#     return jsonify(frequent_set_data)

##### get the system initial prompt and result
@api.route('/load_initial_prompt_and_result', methods = ['GET'])
def load_initial_prompt_and_result():
    initial_prompt = current_app.dataService.load_initial_prompt()
    initial_result = current_app.dataService.generate_results_for_instances(original = True)
    initial_model_performance = current_app.dataService.eval_metric(list(initial_result.keys()), initial_result)
    frequent_set_data = current_app.dataService.reasoning_pattern_mining(initial_result, current_app.dataService.reasoning_cue_embedding, list(initial_result.keys()))
    ###
    # update history prompt
    ###
    history_prompt_num = len(current_app.dataService.history_prompt)
    history_prompt_version = f"Version {history_prompt_num + 1}"
    k_shot_example_num = len(initial_prompt["K_shot_example"])
    current_app.dataService.history_prompt.append({
        "name": history_prompt_version,
        "prompt_text": initial_prompt,
        "accuracy": initial_model_performance,
        "few_shot__num": k_shot_example_num
    })
    return jsonify({'initial_prompt': initial_prompt, 'initial_result': initial_result, 'initial_model_performance': initial_model_performance, 'frequent_set_data': frequent_set_data})


#### get the updated prompt and result (===> acceept submitted prompt from frontend)
@api.route('/load_prompt_and_result', methods = ['POST'])
def load_prompt_and_result():
    data = request.json
    prompt = data["prompt"]
    # prompt = current_app.dataService.current_prompt
    ### here the video list need to consider the extra added test data (the frontend send back)
    videolist = current_app.dataService.valid_data_list
    videolist = current_app.dataService.valid_data_list[0:10]
    print("prompt", prompt)
    result = current_app.dataService.generate_results_for_instances(videolist = videolist, original = False, test_prompt = prompt)
    reasoning_cue_embedding = current_app.dataService.generate_reasoning_cue_embedding(result)
    model_performance = current_app.dataService.eval_metric(videolist, result)
    frequent_set_data = current_app.dataService.reasoning_pattern_mining(result, reasoning_cue_embedding, videolist)
    ###
    # update history prompt
    ###
    history_prompt_num = len(current_app.dataService.history_prompt)
    history_prompt_version = f"Version {history_prompt_num + 1}"
    k_shot_example_num = len(prompt["K_shot_example"]) 
    current_app.dataService.history_prompt.append({
        "name": history_prompt_version,
        "prompt_text": prompt,
        "accuracy": model_performance,
        "few_shot_num": k_shot_example_num
    })
    return jsonify({'prompt': prompt, 'result':result, 'model_performance': model_performance, 'frequent_set_data': frequent_set_data})


## load k-shot example, by default the k-shot is computed on all test instances
@api.route('/load_k_shot_example', methods=['GET'])
def load_k_shot_example(): 
    k_shot_example_list = current_app.dataService.select_k_shot_examples(source_instances = current_app.dataService.valid_data_list, num_k = 5)
    k_shot_example_dict = current_app.dataService.load_k_shot_example_with_reasoning(k_shot_example_list)
    return jsonify({"k_shot_example_dict": k_shot_example_dict, "k_shot_example_list": k_shot_example_list})


### load prompt history 
@api.route('/get_history_prompt', methods = ['GET'])
def get_history_prompt():
    # prompt_history = current_app.dataService.load_history_prompt()
    history_prompt = current_app.dataService.history_prompt
    return jsonify(history_prompt)


### load principle
@api.route('/load_principle_list', methods = ['GET'])
def load_principle_list():
    # prompt_history = current_app.dataService.load_history_prompt()
    principle_list = current_app.dataService.load_principle_list()
    return jsonify({'principle_list': principle_list})

@api.route('/load_projection_pos_data', methods = ['GET'])
def load_projection_pos_data():
    pos_data = current_app.dataService.load_projection_pos_data()
    return jsonify({'pos_data': pos_data})


##### need frontend to send the video list
@api.route('/compute_frequent_sets', methods = ['POST'])
def compute_frequent_sets():
    data = request.json
    video_list = data['video_list']
    # video_list = current_app.dataService.valid_data_list[0:50]
    if len(video_list) == 0:
        return jsonify([])
    result = current_app.dataService.analysis_result
    # print("result", result)
    reasoning_cue_embedding = current_app.dataService.reasoning_cue_embedding
    frequent_set_data = current_app.dataService.reasoning_pattern_mining(result, reasoning_cue_embedding, video_list)
    return jsonify(frequent_set_data)

@api.route('/generate_principle_list_with_instances', methods = ['POST'])
def generate_principle_list():
    data = request.json
    video_list = data['video_list']
    if len(video_list) == 0:
        return jsonify([])
    # low_level_principle_list = current_app.dataService.generate_low_level_principle(video_list, current_app.dataService.analysis_result)
    # high_level_principle_list = current_app.dataService.generate_high_level_principle(low_level_principle_list)
    low_level_principle_dict = current_app.dataService.generate_low_level_principle(video_list, current_app.dataService.analysis_result)
    low_level_principle_list = []
    for key in low_level_principle_dict:
        answer = json.loads(low_level_principle_dict[key])
        low_level_principle_list.append(answer['Insight'])
    high_level_principle_list = current_app.dataService.generate_high_level_principle(list(low_level_principle_dict.values()))

    return jsonify({'low_level_principle_list': low_level_principle_list, 'high_level_principle_list': high_level_principle_list})


@api.route('/load_extra_test_example_with_instances', methods = ['POST'])
def load_extra_test_example_with_instances():
    data = request.json
    source_videos = data['video_list']
    if len(source_videos) == 0:
        return jsonify([])
    retrieved_video_list = current_app.dataService.retrieve_extra_test_data(source_videos = source_videos)
    return jsonify({'retrieved_video_list': retrieved_video_list})



# @api.route('/load_data_for_center_view_table', methods = ['GET'])
# def load_data_for_center_view_table():
#     result = current_app.dataService.generate_results_for_instances(original = True)
#     reasoning_cue_embedding = current_app.dataService.reasoning_cue_embedding
#     frequent_set_data = current_app.dataService.reasoning_pattern_mining(result, reasoning_cue_embedding, list(result.keys()))
#     return jsonify(frequent_set_data)


# @api.route('/load_extra_test_example', methods = ['GET'])
# def load_extra_test_example():
# ### select demonstration example given source videos
# @api.route('/select_demonstration_example', methods=['POST'])
# def select_demonstration_example():
#     data = request.json
#     source_videos = data['sources']
#     # target_videos = data['targets']
#     if len(source_videos) == 0:
#         return jsonify([])
#     k_shot_example_list = current_app.dataService.select_demonstration_example(source_videos)
#     k_shot_example_dict = {}
#     for key in k_shot_example_list:
#         k_shot_example_dict[key] = current_app.dataService.k_shot_example_dict[key] if key in current_app.dataService.k_shot_example_dict else "The reasoning need to be filled manually"
#     return jsonify({"k_shot_example_dict": k_shot_example_dict, "k_shot_example_list": k_shot_example_list})








if __name__ == '__main__':
    pass



