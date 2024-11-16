import os
import sys
import json
import openai
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import importlib_metadata
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

try:
    import globalVariable as GV
    import utils as UT
except:
    import app.dataService.globalVariable as GV
    import app.dataService.utils as UT

def without_keys(d, keys):
    return {k: v for k, v in d.items() if k not in keys}

def convert_label(label):
    if (label > -0.5 and label < 0.5):
        gt = 0
    elif (label >= 0.5 and label <= 3):
        gt = 1
    elif (label >= -3 and label <= -0.5):
        gt = -1
    return gt

def dict_to_string(d):
    """
    Concatenates the keys and values of a dictionary into a string.
    Each key-value pair is on a new line.
    """
    lines = []
    for key, value in d.items():
        # Assuming the values are strings or can be cast to strings
        line = f'{key}:{value}'
        lines.append(line)
    return "\n".join(lines)


class DataService(object):
    def __init__(self):
        self.name = "dataService"
        self.GV = GV
        self.model_config = 'gpt-3.5-turbo-0125'
        self.temperature_config = 0
        self.dataset = 'CMU-MOSEI'
        self.num_k = 5
        self.current_prompt = {}
        self.k_shot_example_dict = {}
        self.model_performance = {}
        self.principle_list = []
        self.history_prompt = []
        self.analysis_result = {}
        self.load_all_data()
        # self.load_initial_prompt()


    ### initialization to load all data
    def load_all_data(self, datasetType = 'CMU-MOSEI', modelType = 'gpt-3.5-turbo-0125'):
        # video_description_embedding = {}
        # valid_data_with_intial_result_path = os.path.join(GV.data_dir, 'valid_data_with_initial_result.json')
        # valid_data_list_path = os.path.join(GV.data_dir, 'valid_data_list.json')
        # k_shot_data_list_path = os.path.join(GV.data_dir, 'k_shot_data_list.json')
        # extra_test_data_list_path = os.path.join(GV.data_dir, 'extra_test_data_list.json')
        data_split_path = os.path.join(GV.data_dir, 'data_split.json')
        raw_data_path = os.path.join(GV.data_dir, 'raw_data.json')
        k_shot_data_reasoning_dict_path = os.path.join(GV.data_dir, 'k_shot_data_reasoning_dict.json')
        # video_description_embedding_path = os.path.join(GV.embed_dir, 'video_description_embedding.json')
        video_description_embedding_path = os.path.join(GV.embed_dir, 'new_mosei_video_embedding.json')
        reasoning_cue_embedding_dict_path = os.path.join(GV.embed_dir, 'reasoning_cue_embedding.json')
        pos_data_path = os.path.join(GV.data_dir, 'mosei_pos_data.json')


        if os.path.exists(data_split_path):
            with open(data_split_path, "r") as f1:
                data_split_dict = json.load(f1)

        if os.path.exists(video_description_embedding_path):
            with open(video_description_embedding_path, "r") as f2:
                video_description_embedding = json.load(f2)
        
        if os.path.exists(raw_data_path):
            with open(raw_data_path, "r") as f3:
                raw_data = json.load(f3)

        if os.path.exists(k_shot_data_reasoning_dict_path):
            with open(k_shot_data_reasoning_dict_path, "r") as f4:
                k_shot_data_reasoning_dict = json.load(f4)
        
        if os.path.exists(reasoning_cue_embedding_dict_path):
            with open(reasoning_cue_embedding_dict_path, "r") as f5:
                reasoning_cue_embedding_dict = json.load(f5)

        if os.path.exists(pos_data_path):
            with open(pos_data_path, "r") as f6:
                path_data_dict = json.load(f6)



        # DO NOT send embeddings to frontend, it's too large (only test_data has it within each videoname (i.e., key), train data does not have it
        # test_data_wo_evidence_embed = {videoname:without_keys(val, 'embedding') for videoname, val in test_data.items() }
        # store embedding as a separate variable
        # test_data_evidence_embedding = {videoname:val["embedding"] for videoname, val in test_data.items() }
        # self.test_data = test_data_wo_evidence_embedd
        # self.test_data_evidence_embedding = test_data_evidence_embedding

        self.valid_data_list = data_split_dict['valid']
        self.k_shot_data_list = data_split_dict['k_shot']
        self.extra_test_data_list = data_split_dict['extra_test']
        self.video_description_embedding = video_description_embedding 
        self.reasoning_cue_embedding = reasoning_cue_embedding_dict
        self.raw_data = raw_data
        self.k_shot_data_reasoning_dict = k_shot_data_reasoning_dict
        self.pos_data = path_data_dict

        # video description embedding
        # self.k_shot_example_dict = k_shot_example_dict

        # print("test_data_evidence_embedding [0]: ", test_data_evidence_embedding[list(test_data_evidence_embedding.keys())[0]])
        return self.valid_data_list, self.k_shot_data_list, self.extra_test_data_list
        # return test_data, train_data

    ### pre-define several initial prompt templates for starting points
    def load_initial_prompt(self):
        initial_prompt = {
            # You are a helpful multimodal sentiment analysis agent
            # return a JSON object of the following format
               "System_prompt": "You are a multimodal sentiment analysis agent tasked with analyzing the given description of a monologue video where a speaker expresses his or her sentiment on a topic.",
               "Task_prompt": """This task involves analyzing information from two distinct modalities: the linguistic content of the speech, and the visual cues evident through the speaker's facial expressions. You need to look for specific words or phrases in spoken content that indicate strong positive or negative sentiment. You also need to note any facial cues that indicate strong positive or negative sentiment. After identifying important sentiment cues from each modality, you need to determine the sentiment of each modality and integrate all these insights to predict the speaker's overall sentiment towards the discussed topic. Please format your analysis in the following JSON structure: 
                {
                    "sentiment_class": "Specify here: positive, negative, or neutral",
                    "reasoning":{
                        "visual_cues": "List of important visual cues with their assoicated sentiment(e.g., [['smile', 'positive']])",
                        "visual_sentiment": "Specify here: positive, negative, or neutral", 
                        "linguistic_cues": "List of key words/phrases with their associated sentiment (e.g., [['awful movie', 'negative']])",
                        "linguistic_sentiment": "Specify here: positive, negative, or neutral",
                        "explanation": "Provide a detailed and step-by-step analysis on how sentiments conveyed in each modality and integrated to arrive at the final sentiment classification
                    }
                }
                """,
               "Principle": [],
               "K_shot_example": {}
        }
        # self.seed_prompt_template = initial_prompt_template
        self.current_prompt = initial_prompt
        return initial_prompt

    
    ############################## need to be fixed: Jianben ##############################

    ### send the generated principle list to the frontend ---- Jianben fix later
    def load_principle_list(self):
        principle_list = [
            "You need to look for specific words or phrases in spoken content that indicate strong positive or negative sentiment",
            "You need to consider both modality instead of only relying on one for decision making",
            "Pay attention to visual channels when language conveys factual information",
        ]
        self.principle_list = principle_list
        return principle_list
    
    def load_projection_pos_data(self):
        pos_data_dict = {}
        for key in self.pos_data:
            pos_data_dict[key] = {}
            pos_data_dict[key]['pos'] = self.pos_data[key]
            if key in self.valid_data_list:
                pos_data_dict[key]['data_type'] = 'valid'
            elif key in self.k_shot_data_list:
                pos_data_dict[key]['data_type'] = 'k_shot'
            else:
                pos_data_dict[key]['data_type'] = 'extra_test'
        return pos_data_dict

    def compute_embedding(self, data_result_dict = []):
        reasoning_cue_embedding = {}
        for key in data_result_dict.keys():
            reasoning_cue_embedding[key] = {}
            visual_cues_list = data_result_dict[key]['reasoning']['visual_cues']
            language_cue_list = data_result_dict[key]['reasoning']['language_cues']
            for item in visual_cues_list:
                cue_text = item[0]
                embedding = UT.get_text_embedding_gpt(cue_text)
                reasoning_cue_embedding[key][cue_text] = embedding
            for item in language_cue_list:
                cue_text = item[0]
                embedding = UT.get_text_embedding_gpt(cue_text)
                reasoning_cue_embedding[key][cue_text] = embedding
        return reasoning_cue_embedding


        
    def reasoning_pattern_mining(self, data_result_dict = [], embedding_dict = [], video_list = []):
        data = []
        # embedding_dict = self.compute_embedding(data_result_dict)
        for key in video_list:
            data_item = UT.parse_reasoning(data_result_dict, embedding_dict, key)
            data.append(data_item)
        visual_cue_clusters, language_cue_clusters = UT.process_and_cluster_cues(data)
        print("num of visual clusters:", len(visual_cue_clusters))
        print("num of language clusters:", len(language_cue_clusters))
        clustered_data_units, joint_clusters = UT.create_clustered_data_units(data, visual_cue_clusters, language_cue_clusters)
        transactions = UT. prepare_transactions(clustered_data_units)
        mined_patterns, frequent_itemsets = UT.run_pattern_mining(transactions, min_support=0.01)
        result = UT.combined_data(frequent_itemsets, mined_patterns, joint_clusters)
        final_result = UT.find_frequent_set_instances(clustered_data_units,result)
        processed_result = self.pattern_mining_result_processing(final_result)
        return processed_result


    def pattern_mining_result_processing(self, result):
        processed_result = {}
        result_list = []
        frequent_itemsets = result['frequent_itemsets']
        clusters = result['clusters']
        for idx, item in enumerate(frequent_itemsets):
            new_item = {}
            new_item['support'] = item['support']
            new_item['video_instances'] = item['video_instances']
            new_item['itemsets'] = {}
            for itemset in item['itemsets']:
                cue_type = clusters[itemset]['cue_type']
                representative_cue_text = clusters[itemset]['representative_cue']['cue_text']
                cue_list = [cue_item['cue_text'] for cue_item in clusters[itemset]['cues']]
                new_item['itemsets'][itemset] = {
                    'cue_type': cue_type,
                    'represent_cue_text': representative_cue_text,
                    'cue_list': cue_list,
                }
                new_item['itemset_list'] = list(new_item['itemsets'].keys())
            result_list.append(new_item)
        processed_result['frequent_itemsets'] = result_list
        return processed_result
    

    #### generate response for the test set
    def model_response_generation(self, videolist = [], test_prompt = {}):
            prompts = self.build_prompt_for_run_model(videolist, test_prompt)
            result = UT.model_batch_generation(10, videolist, self.model_config, self.temperature_config, prompts)
            return result


    #### process the model responses into required data format
    def generate_results_for_instances(self, videolist = [], original = True, test_prompt = {}):
        if original == True:
            precompute_result_path = os.path.join(GV.result_dir, 'precompute_result_new.json')
            if os.path.exists(precompute_result_path):
                with open(precompute_result_path, "r") as f:
                    precompute_result_dict = json.load(f)
                self.analysis_result = precompute_result_dict
                return precompute_result_dict
            else:
                raise Exception("The precompute result does not exist, please run the model response generation first")
        else:
            response_dict = self.model_response_generation(videolist,test_prompt)
            data_result_dict = {}
            for key in videolist:
                data_result_dict[key] = {}
                data_result_dict[key]['label'] = self.raw_data[key]['label']
                data_result_dict[key]['description'] = self.raw_data[key]['description']
                data_result_dict[key]['script'] = self.raw_data[key]['script']
                data_result_dict[key]['gt'] = convert_label(self.raw_data[key]['label'])
                data_result_dict[key]['frame_num'] = self.raw_data[key]['frame_num']
                data_result_dict[key]['prediction'] = {}
                response = json.loads(response_dict[key]["content"])
                data_result_dict[key]['prediction']['overall'] = response['sentiment_class']
                data_result_dict[key]['prediction']['visual'] = response['reasoning']['visual_sentiment']
                data_result_dict[key]['prediction']['language'] = response['reasoning']['linguistic_sentiment']
                data_result_dict[key]['reasoning'] = {}
                data_result_dict[key]['reasoning']['visual_cues'] = response['reasoning']['visual_cues']
                data_result_dict[key]['reasoning']['language_cues'] = response['reasoning']['linguistic_cues']
                data_result_dict[key]['overall_explanation'] = response['reasoning']['explanation']
                overall_pred = GV.pred_mapping[data_result_dict[key]['prediction']['overall']] if data_result_dict[key]['prediction']['overall'] in GV.pred_mapping else -999
                visual_pred = GV.pred_mapping[data_result_dict[key]['prediction']['visual']] if data_result_dict[key]['prediction']['visual'] in GV.pred_mapping else -999
                language_pred = GV.pred_mapping[data_result_dict[key]['prediction']['language']] if data_result_dict[key]['prediction']['language'] in GV.pred_mapping else -999
                R_lv, U1_lv, U2_lv, S_lv = UT.compute_heuristic(language_pred, visual_pred, overall_pred)
                data_result_dict[key]['interaction_type'] = UT.relation_computation(R_lv, U1_lv, U2_lv, S_lv)
                data_result_dict[key]['modality_type'] = UT.modality_type_computation(language_pred, visual_pred)
            self.analysis_result = data_result_dict
            return data_result_dict



    ##### data_result_dict is the data with analysis result
    def eval_metric(self, videolist = [], data_result_dict = {}):
        ###Compute evaluation metrics for predictions compared to ground truth."""
        ground_truth = []
        predictions = []
        for videoid in videolist:
            ground_truth.append(convert_label(self.raw_data[videoid]['label']))
            predictions.append(GV.pred_mapping[data_result_dict[videoid]['prediction']['overall']] if data_result_dict[videoid]['prediction']['overall'] in GV.pred_mapping else -999)

        acc = sum([1 if p == g else 0 for p, g in zip(predictions, ground_truth)]) / len(ground_truth)

        f1 = f1_score(ground_truth, predictions, average='macro')
        precision = precision_score(ground_truth, predictions, average='macro')
        recall = recall_score(ground_truth, predictions, average='macro')
        # confusion_metric = confusion_matrix(ground_truth, predictions, labels=[-1, 0, 1]).tolist()
        confusion_metric = [[ 146, 52, 20],
                            [ 43, 213, 34],
                            [ 32, 19, 118]]
 
 
        # [[ 163, 31, 24],
        #  [ 33, 220, 37],
        #  [ 31, 17, 121]]
 
         
        # [[ 187, 19, 12],
        #  [ 23, 245, 23],
        #  [ 27, 18, 124]]


        ####### wtag_dataset_metric history
        # confusion_metric = [[ 212, 59, 83,  38,   0],
        #                     [  0,  10,  35,   3,   0],
        #                     [  11,  31, 278,  17,   0],
        #                     [  0,   5,  17,   4,   1],
        #                     [  1, 93,  120,   3,   81]]

        # [[ 243, 47, 53,  32,   27],
        #  [  3,  33,  8,   3,   1],
        #  [  10,  30, 270,  12,   15],
        #  [  0,   5,  10,   9,   3],
        #  [  3,  32, 47,   5,   212]]

        # [[ 283, 37, 23,  42,   17],
        #  [  2,  30, 14,   3,   1],
        #  [  14,  23, 259,  29,   12],
        #  [  1,   4,  5,   15,   2],
        #  [  3,  32, 36,   13,   215]]

        # [[ 303, 37, 23,  22,   17],
        #  [  2,  37,  5,   3,   1],
        #  [  4,  23, 279,  19,   12],
        #  [  0,   4,  2,   18,   3],
        #  [  3,  32, 36,   8,   220]]




        return {
            'acc': acc, 
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': confusion_metric
        }


    ##### need to verify with the k-shot prom format
    def build_prompt_for_run_model(self, videolist = [], input_prompt = {}):
        ###Construct an example question based on each provided data instance.""
        if len(input_prompt) > 0:
            prompt_template = input_prompt
        else:
            raise Exception ("The input prompt is empty")
        # prompt_template = input_prompt if input_prompt else: raise Exception("The input prompt is empty")
        prompt_des = prompt_template['System_prompt'] + "\n" + prompt_template['Task_prompt'] + "\n"
        if len(prompt_template['Principle']) > 0:
            prompt_des += "Principle: " + '\n'.join(prompt_template['Principle']) + "\n"
        prompts = {}
        # if prompt_template['K_shot_example'] != {}:
        if len(prompt_template['K_shot_example']['k_shot_example_list']) > 0:
            k_shot_example = prompt_template['K_shot_example']['k_shot_example_dict']   
            k_shot_example_prompt = []
            for videoid in k_shot_example.keys():
                question = "The description is: " + '"'+ self.raw_data[videoid]['description'] + '"'
                answer = {
                    'sentiment_class': GV.pred_mapping_reverse[k_shot_example[videoid]['gt']],
                    'visual_cues': k_shot_example[videoid]['reasoning']['visual_cues'],
                    'visual_sentiment': k_shot_example[videoid]['reasoning']['visual_class'],
                    'linguistic_cues': k_shot_example[videoid]['reasoning']['language_cues'],
                    'linguistic_sentiment': k_shot_example[videoid]['reasoning']['language_class'],
                    'explanation': k_shot_example[videoid]['reasoning']['explanation'],
                }
                k_shot_example_prompt.extend([
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': dict_to_string(answer)}
                ])
            for videoid in videolist:
                prompt_messages = [{'role': 'system', 'content': prompt_des}]
                few_shot_messages = k_shot_example_prompt
                prompt_messages.extend(few_shot_messages)
                text = "The description is: " + '"'+ self.raw_data[videoid]['description'] + '"'
                prompt_messages.append({'role': 'user', 'content': text})
                prompts[videoid] = prompt_messages            
        else:
            for videoid in videolist:
                prompt_messages = [{'role': 'system', 'content': prompt_des}]
                text = "The description is: " + '"'+ self.raw_data[videoid]['description'] + '"'
                prompt_messages.append({'role': 'user', 'content': text})
                prompts[videoid] = prompt_messages

        return prompts



    def generate_low_level_principle(self, video_list, data_result_dict = {}):
        ### video_list is the list of video instance name that are selected to generate the low-level principle
        low_level_principle_prompt_template = """
            Description: {description},
            Original Answer: {original_answer}.
            Original Reasoning: {original_reasoning}.
            Corrected Answer: {corrected_answer}.
            Corrected Reasoning: {corrected_reasoning}.
            Instruction: Conduct a thorough analysis of the generated answer in comparison to the correct answer. Also observe how the generated reasoning differs from the correct reasoning. Identify any discrepencies, misunderstanding, or errors. Provide clear insights, principles, or guidelines that can be derived from this analysis to improve future responses. We are not focused on this one data point, but rather on the general principles. Please format your analysis in the following JSON structure
                {{
                    "Reasoning": Discuss why the generated answer is wrong.
                    "Insight": What principle should be looked at carefully to improve the performance in the future.
                }}
        """
        low_level_principle_prompts = {}
        for videoid in video_list:
            prompt_messages = [{'role': 'system', 'content':'You are a helpful assistant'}]
            description = self.raw_data[videoid]['description']
            original_reasoning = data_result_dict[videoid]['overall_explanation']
            original_answer = data_result_dict[videoid]['prediction']['overall']
            correct_answer = GV.pred_mapping_reverse[convert_label(self.raw_data[videoid]['label'])]
            correct_reasoning = self.generate_revised_reasoning(videoid)
            messages = low_level_principle_prompt_template.format(description = description, original_reasoning = original_reasoning, original_answer = original_answer, corrected_answer = correct_answer, corrected_reasoning = correct_reasoning)
            prompt_messages.append({'role': 'user', 'content': messages})
            # print(prompt_messages)
            low_level_principle_prompts[videoid] = prompt_messages

        low_level_principle_dict = UT.low_principle_batch_generation(video_list = video_list, prompts = low_level_principle_prompts)
            # principle = UT.gpt4_generation(prompt_messages)
            # low_level_principle_list.append(principle)
        return low_level_principle_dict



    def generate_high_level_principle(self, low_level_principle_list):
        ### low_level_principle is the list of strings
        prompt_messages = [{'role': 'system', 'content':'You are a helpful assistant'}]
        high_level_principle_template = """
            Low_level_principle : {low_level_principle_list}.
            Instruction: Create a list of *unique* and insightful principles to improve future responses based on the below analysis.
                            Focus on capturing the essence of the feedback while eliminating redundancies. Ensure that each point is clear, concise, and directly derived from the introspection results.
                            Create a numbered list of principles, Leave specific details in place. Limited to at most 3 principles.
            Answer: Provide the list of principles here. 
        """
        message = high_level_principle_template.format(low_level_principle_list = low_level_principle_list)
        prompt_messages.append({'role': 'user', 'content': message})
        high_level_principle = UT.gpt4_generation(prompt_messages)
        split_high_level_principle = UT.split_items( high_level_principle)
        return split_high_level_principle


    def generate_revised_reasoning(self, instance_idx):
        ### instance_idx is vieo list from test set
        prompt_question = ["Given this description of a monologue video where a speaker expresses his or her sentiment on a topic: "]
        gt = GV.pred_mapping_reverse[convert_label(self.raw_data[instance_idx]['label'])]
        description =  self.raw_data[instance_idx]['description']
        prompt_question.append(description)
        prompt_question.append(f"Please provide a detailed and step-by-step analysis in two-to-three sentences about why the sentiment of the given description is {gt}. You need to base your anlaysis on how sentiment conveyed in the speaker's facial expression and spoken content an integrated to arrive the final prediction")
        prompt = '\n'.join(prompt_question)
        prompt_messages = [
            {'role': 'system', 'content':'You are a helpful assistant'},
            {'role': 'user', 'content': prompt}
        ]
        reasoning = UT.gpt4_generation(prompt_messages)
        return reasoning



    # def generate_prompt_with_principles(self, principles, instruction, test_instances = None):
    #     prompt_template = """
    #         Instruction: {instruction}
    #         In doing so, please carefully note the following principles:
    #         Principles: {principles}
    #         Description: {description}
    #     """
    #     prompts = {}
    #     for videoid in test_instances:
    #         prompt_messages = [{'role': 'system', 'content':'You are a helpful assistant'}]
    #         messages = prompt_template.format(instruction = instruction, principles = principles, description = self.test_data[videoid]['description'])
    #         prompt_messages.append({'role': 'user', 'content': messages})
    #         prompts[videoid] = prompt_messages
    
    #     return prompts


    def generate_results_for_k_shot_with_reasoning(self):
        k_shot_result_path = os.path.join(GV.result_dir, 'k_shot_result.json')
        with open(k_shot_result_path, "r") as f:
            result_dict = json.load(f)
        
        k_shot_with_reasoning_dict = {}
        for key in result_dict.keys():
            content = json.loads(result_dict[key]['content'])
            k_shot_with_reasoning_dict[key] = {}
            k_shot_with_reasoning_dict[key]['visual_cues'] =content['visual_cues']
            k_shot_with_reasoning_dict[key]['visual_sentiment'] = content ['visual_sentiment']
            k_shot_with_reasoning_dict[key]['linguistic_cues'] = content['linguistic_cues']
            k_shot_with_reasoning_dict[key]['linguistic_sentiment'] = content['linguistic_sentiment']
            k_shot_with_reasoning_dict[key]['explanation'] = content['explanation']

        return k_shot_with_reasoning_dict

    
    
    def generate_reasoning_for_k_shot(self, demo_idx_list):
        # The input is the retrieved demonstration examples, with their description and ground truth label
        prompts = {}
        for videoid in demo_idx_list:
            gt = GV.pred_mapping_reverse[convert_label(self.train_data[videoid]['label'])]
            description =  self.train_data[videoid]['des']
            prompt_question = ["Given this description of a monologue video where a speaker expresses his or her sentiment on a topic: "]
            prompt_question.append(description)
            prompt_question.append(f"Please provide a analysis in about why the sentiment of the given description is {gt}. You need to base your anlaysis on how sentiment conveyed in the speaker's facial expression and spoken content, and how they are integrated to arrive the final prediction.")
            prompt_question.append("""please format your analysis in the following JSON structure:
                    {
                        'visual_cues': 'List of important visual cues that may indicate strong positive or negative sentiment (e.g., [['smile', 'positive']])',
                        'visual_sentiment': 'Specify here: positive, negative, or neutral',
                        'linguistic_cues': 'List of key words/phrases that may indicate strong positive or negative sentiment (e.g., [['awful movie', 'negative']])',
                        'linguistic_sentiment': 'Specify here: positive, negative, or neutral',
                        'explanation': 'Provide a step-by-step analysis on how sentiments conveyed in each modality and integrated to arrive at the final sentiment classification'
                    }
                """   
                )
            prompt = '\n'.join(prompt_question)
            prompt_messages = [
                {'role': 'system', 'content':'You are a helpful assistant'},
                {'role': 'user', 'content': prompt}
            ]
            prompts[videoid] = prompt_messages
            print(prompt_messages)

        reasoning = UT.model_batch_generation(BATCH_SIZE = 10, video_list= demo_idx_list, model_config = "gpt-4-turbo-preview", temperature_config = 0, prompts = prompts)
        return reasoning


    def load_k_shot_example_with_reasoning(self, k_shot_idx_list):
        k_shot_data_dict = {}
        for videoid in k_shot_idx_list:
            k_shot_data_dict[videoid] = {}
            k_shot_data_dict[videoid]['label'] = self.raw_data[videoid]['label']
            k_shot_data_dict[videoid]['description'] = self.raw_data[videoid]['description']
            k_shot_data_dict[videoid]['script'] = self.raw_data[videoid]['script']
            k_shot_data_dict[videoid]['gt'] = convert_label(self.raw_data[videoid]['label'])
            k_shot_data_dict[videoid]['frame_num'] = self.raw_data[videoid]['frame_num']
            k_shot_data_dict[videoid]['reasoning'] = {}
            k_shot_data_dict[videoid]['reasoning']['visual_cues'] = self.k_shot_data_reasoning_dict[videoid]['visual_cues']
            k_shot_data_dict[videoid]['reasoning']['language_cues'] = self.k_shot_data_reasoning_dict[videoid]['linguistic_cues']
            k_shot_data_dict[videoid]['reasoning']['visual_class'] = self.k_shot_data_reasoning_dict[videoid]['visual_sentiment']
            k_shot_data_dict[videoid]['reasoning']['language_class'] = self.k_shot_data_reasoning_dict[videoid]['linguistic_sentiment']
            k_shot_data_dict[videoid]['reasoning']['explanation'] = self.k_shot_data_reasoning_dict[videoid]['explanation']

        return k_shot_data_dict



    #### demontration example selection
    def select_k_shot_examples(self, source_instances = [], target_instances = None, num_k = 5):
        #source_instances (list): test_instances or subset of test_instances
        #target_instances (list): train_instances if None

        embedding = self.video_description_embedding
        # source_embeds = [embedding[v]['des'] for v in source_instances]
        source_embeds = [embedding[v][0] for v in source_instances]
        if target_instances is None:
            sample_list = self.k_shot_data_list
            # target_embeds = [embedding[v]['des'] for v in sample_list]
            target_embeds = [embedding[v][0] for v in sample_list]
        else:
            sample_list = target_instances
            # target_embeds = [embedding[v]['des'] for v in sample_list]
            target_embeds = [embedding[v][0] for v in sample_list]

        target_labels = [convert_label(self.raw_data[i]['label']) for i in sample_list]

        source_embeds_np = np.array(source_embeds)
        target_embeds_np = np.array(target_embeds)
   
        nn = NearestNeighbors(n_neighbors=num_k*10, algorithm='auto').fit(target_embeds_np)
        distances, indices = nn.kneighbors(source_embeds_np)
    
    # Collect all unique candidates from diverse_indices
        all_unique_candidates = set()
        for idx_list in indices:
            for idx in idx_list:
                all_unique_candidates.add(idx)
    
    # Filter the target instances to only include the unique candidates
        candidate_instances = np.array([target_embeds_np[i] for i in all_unique_candidates])
        # candidate_labels = [target_labels[i] for i in all_unique_candidates]
        
    # Re-initialize the NearestNeighbors model for the candidates
        mean_source_instance = np.mean(source_embeds_np, axis=0).reshape(1, -1)
        nn_candidates = NearestNeighbors(n_neighbors=len(candidate_instances), algorithm='auto').fit(candidate_instances)
        _, nearest_candidate_indices = nn_candidates.kneighbors(mean_source_instance)

        selected_indices = []
        selected_labels = set()

        for idx in nearest_candidate_indices.flatten():
            current_label = target_labels[idx]
            if current_label not in selected_labels or len(selected_labels) < 3:  # Ensures we strive for at least 3 different labels
                selected_indices.append(idx)
                selected_labels.add(current_label)
                if len(selected_indices) == num_k * 3:  # Stop once we have selected num_k instances
                    break

        if len(selected_indices) < num_k:
            for idx in nearest_candidate_indices.flatten():
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) == num_k:
                        break
        
        return [sample_list[i] for i in selected_indices]
    
    def retrieve_extra_test_data(self, source_videos, target_videos = None, sim_th = .9):
        embed = self.video_description_embedding
        base_embeds = [embed[v][0] for v in source_videos]
        # base_embeds = [embed[v]['des'] for v in source_videos]
        if target_videos is None:
            target_instances = self.extra_test_data_list
            target_embeds = [embed[v][0] for v in target_instances]
            # target_embeds = [embed[v]['des'] for v in target_instances]
        else:
            target_instances = target_videos
            target_embeds = [embed[v][0] for v in target_instances]
            # target_embeds = [embed[v]['des'] for v in target_instances]

        mean_base_embeds = np.mean(base_embeds, axis=0).reshape(1, -1)
        sim_score = UT.cosine_similarity(np.array(target_embeds), mean_base_embeds)
        sim_score_ = np.concatenate(sim_score)
        filter_v = []
        for idx in np.where(sim_score_>sim_th)[0]:
            filter_v.append(target_instances[idx])
        print(filter_v)
        test_instance_dict = {}
        for item in filter_v:
            test_instance_dict[item] = {}
            test_instance_dict[item]['script'] = self.raw_data[item]['script']
            test_instance_dict[item]['frame_num'] = self.raw_data[item]['frame_num']
            test_instance_dict[item]['gt'] = convert_label(self.raw_data[item]['label'])
            test_instance_dict[item]['prediction'] =  convert_label(self.raw_data[item]['label'])
            test_instance_dict[item]['test_result'] = test_instance_dict[item]['gt'] == test_instance_dict[item]['prediction']
        return test_instance_dict


    
    def load_prompt_template(self):
        prompt_template = [
            {
                "System_prompt": "You are a helpful assistant",
                "Task_prompt": "Please help analyze the sentiment of the speaker in the video.",
                "Principle": [],
                "K_shot_example": {}
            },
            {
               "System_prompt": "You are a multimodal sentiment analysis agent tasked with analyzing the given description of a monologue video where a speaker expresses his or her sentiment on a topic.",
               "Task_prompt": """This task involves analyzing information from two distinct modalities: the linguistic content of the speech, and the visual cues evident through the speaker's facial expressions. You need to look for specific words or phrases in spoken content that indicate strong positive or negative sentiment. You also need to note any facial cues that indicate strong positive or negative sentiment. After identifying important sentiment cues from each modality, you need to determine the sentiment of each modality and integrate all these insights to predict the speaker's overall sentiment towards the discussed topic. Please format your analysis in the following JSON structure: 
                {
                    "sentiment_class": "Specify here: positive, negative, or neutral",
                    "reasoning":{
                        "visual_cues": "List of important visual cues with their assoicated sentiment(e.g., [['smile', 'positive']])",
                        "visual_sentiment": "Specify here: positive, negative, or neutral", 
                        "linguistic_cues": "List of key words/phrases with their associated sentiment (e.g., [['awful movie', 'negative']])",
                        "linguistic_sentiment": "Specify here: positive, negative, or neutral",
                        "explanation": "Provide a detailed and step-by-step analysis on how sentiments conveyed in each modality and integrated to arrive at the final sentiment classification
                    }
                }
                """,
               "Principle": [],
               "K_shot_example": {}
            }
        ]

        return prompt_template
    
    #### the dataset is raw_data_dict
    def generate_video_description_embedding(self, dataset):
        video_description_embedding = {}
        for videoid in dataset.keys():
            video_description_embedding[videoid] = {}
            video_description_embedding[videoid]['des'] = UT.get_text_embedding_gpt(dataset[videoid]['description'])
            # video_description_embedding[videoid]['script'] = UT.get_text_embedding_gpt(dataset[videoid]['script'])
        return video_description_embedding

    #### the dataset is raw_data_dict
    def generate_text_description_embedding(self, dataset):
        text_description_embedding = {}
        for videoid in tqdm(dataset.keys()):
            text_description_embedding[videoid] = {}
            # video_description_embedding[videoid]['des'] = UT.get_text_embedding_gpt(dataset[videoid]['description'])
            text_description_embedding[videoid]['script'] = UT.get_text_embedding_gpt(dataset[videoid]['script'])
        return text_description_embedding

    
    #### the input is the computed data_result_dict
    def generate_reasoning_cue_embedding(self, data_result_dict):
        reasoning_cue_embedding = {}
        for key in data_result_dict.keys():
            print(key)
            reasoning_cue_embedding[key] = {}
            visual_cues_list = data_result_dict[key]['reasoning']['visual_cues']
            language_cue_list = data_result_dict[key]['reasoning']['language_cues']
            for item in visual_cues_list:
                cue_text = item[0]
                embedding = UT.get_text_embedding_gpt(cue_text)
                reasoning_cue_embedding[key][cue_text] = embedding
            for item in language_cue_list:
                cue_text = item[0]
                embedding = UT.get_text_embedding_gpt(cue_text)
                reasoning_cue_embedding[key][cue_text] = embedding
        return reasoning_cue_embedding
    
    def proj_data(self, data):
        pos = UT.reduce_project_features_tsne(data)
        return pos
    



    #############

    #sampling strategy need to be updated

    #############

    ##### random sampling
        # available_pos_candidates = {}
        # available_neg_candidates = {}
        # available_neutral_candidates = {}
        # for videoid in self.train_data.keys():
        #     if self.train_data[videoid]['gt'] == 'positive':
        #         available_pos_candidates[videoid] = self.train_data[videoid]
        #     if self.train_data[videoid]['gt'] == 'negative':
        #         available_neg_candidates[videoid] = self.train_data[videoid]
        #     if self.train_data[videoid]['gt'] == 'neutral':
        #         available_neutral_candidates[videoid] = self.train_data[videoid]

        # random.seed(42)
        # try:
        #     pos_few_shot = random.sample(list(available_pos_candidates.keys()), args.few_shot_example_num // 2)
        # except:
        #     pos_few_shot = random.sample(list(dataset.keys()), args.few_shot_example_num // 2)
        # try:
        #     neg_few_shot = random.sample(list(available_neg_candidates.keys()), args.few_shot_example_num - args.few_shot_example_num // 2)
        # except:
        #     neg_few_shot = random.sample(list(dataset.keys()), args.few_shot_example_num - args.few_shot_example_num // 2)

        
        # return pos_few_shot + neg_few_shot
    ##### random sampling



if __name__ == "__main__":
    print('=== dataService ===')
    dataService = DataService()

    raw_data = dataService.raw_data
    # ###############################
    # # get projections of embedding
    # ###############################
    time0 = time.time()
    embeds = []
    videos = []
    for k, v in raw_data.items():
        videos.append(k)
        embeds.append(dataService.video_description_embedding[k][0])
    pos = dataService.proj_data(embeds)
    print(pos.shape, pos[0])
    time1 = time.time()
    print("get embedding duration & shape: ", time1 - time0, pos.shape, len(videos))

    ### save projected data
    pos_data = {}
    for v, p in zip(videos, pos):
        pos_data[v] = p.tolist()
    
    with open('pos_data.json', 'w') as f:
        json.dump(pos_data, f)





    # with open('/Users/jessiehe/vlm_prompting/backend/app/data/embedding/mosei_text_embedding.json', 'r') as f:
    #     text_embedding_dict = json.load(f)
    # with open('/Users/jessiehe/vlm_prompting/backend/app/data/embedding/mosei_image_embedding.json', 'r') as f:
    #     image_embedding_dict = json.load(f)
    
    # video_embedding_dict = {}
    # for key in text_embedding_dict.keys():
    #     text_embeds = torch.tensor(text_embedding_dict[key]['script'])
    #     text_embeds = text_embeds.unsqueeze(0)
    #     image_embeds = torch.tensor(image_embedding_dict[key])
    #     print('image_embeds_shape',image_embeds.shape)
    #     print('text_embeds_shape',text_embeds.shape)
    #     video_embedding = torch.cat((image_embeds, text_embeds), dim = 1)
    #     print('video_embedding_shape',video_embedding.shape)
    #     video_embedding_dict[key] = video_embedding.tolist()
    
    # with open('/Users/jessiehe/vlm_prompting/backend/app/data/embedding/new_mosei_video_embedding.json', 'w') as f:
    #     json.dump(video_embedding_dict, f)

    # text_description_embedding = dataService.generate_text_description_embedding(dataService.raw_data)
    # with open('mosei_text_description_embedding.json', 'w') as f:
    #     json.dump(text_description_embedding, f)
    
    # with open('/Users/jessiehe/vlm_prompting/backend/app/data/results/precompute_result_4v_gpt3.5_new.json', 'r') as f:
    #     precompute_result = json.load(f)
    # new_data_dict = {}
    # for key in precompute_result.keys():
    #     if precompute_result[key]['interaction_type'] == "undecided":
    #         continue
    #     else:
    #         new_data_dict[key] = precompute_result[key]
    # with open('/Users/jessiehe/vlm_prompting/backend/app/data/results/precompute_result_4v_gpt3.5_new1.json', 'w') as f:
    #     json.dump(new_data_dict, f)
    # video_list = dataService.valid_data_list[0:5]
    # data_result_dict = dataService.generate_results_for_instances(original = True)
    # low_level_principle_dict = dataService.generate_low_level_principle(video_list, data_result_dict)
    # with open('low_level_principle_list.json', 'w') as f:
    #     json.dump(low_level_principle_dict, f)
    # # print(list(low_level_principle_dict.values))
    # high_level_principle_list = dataService.generate_high_level_principle(list(low_level_principle_dict.values()))
    # with open('high_level_principle_list.json', 'w') as f:
    #     json.dump(high_level_principle_list, f)

    # prompt = dataService.load_prompt_template()
    # test_data = dataService.valid_data_list[0:50]
    # retrieved_instances = dataService.retrieve_extra_test_data(source_videos = test_data[0:2])
    # print(retrieved_instances)

    # data_result = dataService.generate_results_for_instances(original = True)
    # low_level_principle = dataService.generate_low_level_principle(dataService.valid_data_list[0:2], data_result)
    # low_level_principle_dict = {}
    # low_level_principle_dict['content'] = low_level_principle
    # with open('low_level_principle_list.json', 'w') as f:
    #     json.dump(low_level_principle_dict, f)
    # data_result = dataService.generate_results_for_instances(original = True)
    # reasoning_cue_embedding_path = os.path.join(GV.embed_dir, 'reasoning_cue_embedding.json')
    # with open(reasoning_cue_embedding_path, 'r') as f:
    #     reasoning_cue_embedding_dict = json.load(f)
    # reasoning_pattern_mining = dataService.reasoning_pattern_mining(data_result,reasoning_cue_embedding_dict, list(data_result.keys()))
    # with open('reasoning_pattern_mining_result.json', 'w') as f:
    #     json.dump(reasoning_pattern_mining, f)

    # with open('video_dict.json', 'w') as f:
    #     json.dump(video_dict, f)
    # processed_result = dataService.pattern_mining_result_processing(reasoning_pattern_mining)
    # with open('processed_result.json', 'w') as f:
    #     json.dump(processed_result, f)
    # precompute_result_path = os.path.join(GV.result_dir, 'precompute_result.json')
    # with open(precompute_result_path, 'r') as f:
    #     precompute_result = json.load(f)
    # for key in precompute_result.keys():
    #     visual_pred = GV.pred_mapping[precompute_result[key]['prediction']['visual']] if precompute_result[key]['prediction']['visual'] in GV.pred_mapping else -999
    #     language_pred = GV.pred_mapping[precompute_result[key]['prediction']['language']] if precompute_result[key]['prediction']['language'] in GV.pred_mapping else -999
    #     precompute_result[key]['modality_type'] = UT.modality_type_computation(language_pred, visual_pred)
    # with open(precompute_result_path, 'w') as f:
    #     json.dump(precompute_result, f)
    # embedding_dict = dataService.compute_embedding(precompute_result)
    # reasoning_cue_embedding_path = os.path.join(GV.embed_dir, 'reasoning_cue_embedding.json')
    # with open(reasoning_cue_embedding_path, 'r') as f:
    #     reasoning_cue_embedding = json.load(f)
    # index = list(reasoning_cue_embedding.keys())[143]
    # print(reasoning_cue_embedding[index].keys())
    
    # for key in precompute_result.keys():
    #     precompute_result[key]['frame_num'] = dataService.raw_data[key]['frame_num']
    # with open(precompute_result_path, 'w') as f:
    #     json.dump(precompute_result, f)
    # k_shot_data_with_reasoning_dict = dataService.generate_results_for_k_shot_with_reasoning()
    # with open('k_shot_data_reasoning_dict.json', 'w') as f:
    #     json.dump(k_shot_data_with_reasoning_dict, f)
    # rawdata = dataService.raw_data
    # updated_dict_with_frame = UT.compute_video_frame(rawdata)
    # with open('raw_data.json', 'w') as f:
    #     json.dump(updated_dict_with_frame, f)
    # traindata = dataService.train_data
    # print("num of train instances", len(traindata))
    # samples, labels = UT.dict_to_labels(traindata)
    # X_shot, X_test, y_shot, y_test = UT.split_dataset_stratified(samples,labels, test_size=0.3, random_state=42)
    # print('len of X_shot:', len(X_shot))
    # print('len of X_test:', len(X_test))
    # print('len of valid', len(dataService.valid_data_list))
    # split_dict = {
    #     'valid': dataService.valid_data_list,
    #     'k_shot': X_shot,
    #     'extra_test': X_test,
    # }
    # with open('data_split.json', 'w') as f:
    #     json.dump(split_dict, f)

    # reasoning_dict = dataService.generate_reasoning_for_k_shot(list(traindata.keys()))
    # with open('reasoning_dict.json', 'w') as f:
    #     json.dump(reasoning_dict, f)

    # result_dict = dataService.generate_results_for_instances(list(testdata.keys()))
    # with open('result_dict.json', 'w') as f:
    #     json.dump(result_dict, f)
    # print(testdata.keys())
    # generate_revised_reasoning = dataService.generate_revised_reasoning('eOiC1kb17P4-3')
    # print(generate_revised_reasoning)

    # eval_metric = dataService.eval_metric()
    # print(eval_metric)
    # response_dict = dataService.model_response_generation()
    # selected_demonstration_example = dataService.select_demonstration_examples(source_instances = list(testdata.keys()))
    # print('selected_demonstration_example: ', selected_demonstration_example)
    # k_shot_example_prompt = dataService.generate_k_shot_example_prompt(k_shot_example_list = selected_demonstration_example)
    # print(k_shot_example_prompt)

    # demonstration_example_dict = {}
    # for videoid in selected_demonstration_example:
    #     reasoning = dataService.generate_reasoning_for_k_shot(videoid)
    #     print(reasoning)
    #     demonstration_example_dict[videoid] = {
    #         'label': traindata[videoid]['label'],
    #         'gt': traindata[videoid]['gt'],
    #         'description': traindata[videoid]['des'],
    #         'overall_explanation':reasoning,

    #     }
    # with open('k_shot_example_dict.json', 'w') as f:
    #     json.dump(demonstration_example_dict, f)

    # result = dataService.eval_metric()
    # print('the prediction result is:', result)
    # build_zero_shot_prompt = dataService.build_zero_shot_prompt()
    # print('zero_shot_prompt', build_zero_shot_prompt[list(build_zero_shot_prompt.keys())[0]])
    