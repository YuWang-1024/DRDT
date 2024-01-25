import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
from recbole.model.abstract_recommender import SequentialRecommender
import json
from openai import OpenAI


class LLM_GPT(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        
        self.device = "cuda"
        api_key = config['api_key']
        self.api_model_name = config['api_name']
        self.model_path = config['model_path']      
        self.max_tokens = config['max_tokens']
        self.temperature = config['temperature']
        self.max_his_len = config['max_his_len']
        
        self.recall_budget = config['recall_budget']
        self.boots = config['boots']
        
        self.data_path = config['data_path']
        self.dataset_name = dataset.dataset_name
        self.id_token = dataset.field2id_token['item_id'] # list of item tokens (token means the original user IDs), and the position is corresponding to the item IDs.
        self.item_text = self.load_text() # list of item titles where 1:1 map the id_token list.
        self.logger.info(f'Avg. t = {np.mean([len(_) for _ in self.item_text])}')
        self.fake_fn = torch.nn.Linear(1, 1)
        
        
        self.model_version = config['model_version']
        self.model_name = 'gpt-4'
        self.model_name_long = 'gpt-4-32k'
        
        print(f"model_name: {self.model_name}")
        self.reflex_step = config['reflex_step']
        self.client = OpenAI(api_key=api_key)
        
        


    def load_text(self):
        token_text = {} # dictionary that maps item tokens to their corresponding title
        item_text = ['[PAD]']
        feat_path = osp.join(self.data_path, f'{self.dataset_name}.item')
        if self.dataset_name == 'ml-1m':
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, movie_title, release_year, genre = line.strip().split('\t')
                    token_text[item_id] = movie_title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                if raw_text.endswith(', The'):
                    raw_text = 'The ' + raw_text[:-5]
                elif raw_text.endswith(', A'):
                    raw_text = 'A ' + raw_text[:-3]
                item_text.append(raw_text)
            return item_text
        elif self.dataset_name == 'Games' or self.dataset_name == 'Books' or self.dataset_name == 'Magazine' or self.dataset_name == 'Luxury':
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, title = line.strip().split('\t')
                    token_text[item_id] = title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                item_text.append(raw_text)
            return item_text
        else:
            raise NotImplementedError()

    def predict_on_subsets(self, interaction, idxs, total_examples):
        """
        Main function to rank with LLMs

        :param interaction:
        :param idxs: item id retrieved by candidate generation models [batch_size, candidate_size]
        :return:
        """

        batch_size = idxs.shape[0]
        answer_list = {}
        scores = torch.full((idxs.shape[0], self.n_items), -10000.).to('cpu')
        

        for i in tqdm(range(batch_size)):
            user_his_text, candidate_text, candidate_text_order, candidate_idx, ground_truth_text = self.get_batch_inputs(interaction, idxs, i)
            
            if self.model_version == 'COT':
                results, message_content = self.inference_COT(self.dataset_name, user_his_text, candidate_text_order)
                output = results['rank']
                prompt = message_content
            
            elif self.model_version == 'COT_example':
                example_text = self.get_batch_examples(total_examples, i)
                results, message_content = self.inference_COT_example(self.dataset_name, user_his_text, candidate_text_order, example_text[0])
                output = results['rank']
                prompt = message_content

            
            elif self.model_version == 'COT_reflex':
                example_text = self.get_batch_examples(total_examples, i)
                results, message_content = self.inference_COT_reflex(self.dataset_name, user_his_text, candidate_text_order, example_text[0])
                output = results['rank']
                prompt = message_content

                
            elif self.model_version == 'COT_reflex_long':
                example_text = self.get_batch_examples(total_examples, i)
                results, message_content = self.inference_COT_reflex_long(self.dataset_name, user_his_text, candidate_text_order, example_text[0])
                output = results['rank']
                prompt = message_content
            
            elif self.model_version== 'COT_reflex_NE': # no example
                results, message_content = self.inference_COT_reflex(self.dataset_name, user_his_text, candidate_text_order, None)
                output = results['rank']
                prompt = message_content
            elif self.model_version == 'COT_reflex_long_NE':
                results, message_content = self.inference_COT_reflex_long(self.dataset_name, user_his_text, candidate_text_order, None)
                output = results['rank']
                prompt = message_content

            
            elif self.model_version=='Ori': # without ICL

                # version without in context examples
                prompt = self.construct_prompt(self.dataset_name, user_his_text, candidate_text_order)
                messages_content = [
                    {'role': "system", "content": "You are a helpful recommender system"}
                ]
                messages_content.append({'role': "user", "content": prompt})
                output = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages_content
                ).choices[0].message.content
            
            elif self.model_version=='ICL_pre': # ICL from llm_rank that use one-step previous as icl example
                prompt = self.construct_prompt_ICL_pre(self.dataset_name, user_his_text, candidate_text_order)
                messages_content = [
                    {'role': "system", "content": "You are a helpful recommender system"}
                ]
                messages_content.append({'role': "user", "content": prompt})
                output = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages_content
                ).choices[0].message.content

                
            elif self.model_version == 'ICL': # use the semantic retrieve sequence as icl example
                                
                example_text = self.get_batch_examples(total_examples, i)
                prompt = self.construct_prompt_ICL(self.dataset_name, user_his_text, candidate_text_order, example_text)
                messages_content = [
                    {'role': "system", "content": "You are a helpful recommender system"}
                ]
                messages_content.append({'role': "user", "content": prompt})
                output = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages_content
                ).choices[0].message.content
                
            elif self.model_version == 'COT_Ori':
                prompt = self.construct_prompt_COT_Ori(self.dataset_name, user_his_text, candidate_text_order)
                messages_content = [
                    {'role': "system", "content": "You are a helpful recommender system"}
                ]
                messages_content.append({'role': "user", "content": prompt})
                output = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages_content
                ).choices[0].message.content
                
                    
            else:
                assert "NotImplementedError"
                

            self.parsing_output_text(scores, i, output)
            print(f"*********{i}*********")
            print(f"prompt: {prompt}\n")
            print(f"answer: {output}\n")
            print(f"Ground truth: {ground_truth_text}\n")
            answer_list[i] = { "prompt": prompt, "response": output, "ground_truth": ground_truth_text}
        with open(f'results/{self.dataset_name}_{self.model_name}_{self.model_version}_{self.reflex_step}.json','w') as file:
            json.dump(answer_list, file)
        return scores


    def get_batch_inputs(self, interaction, idxs, i):

        user_his = interaction[self.ITEM_SEQ]
        user_his_len = interaction[self.ITEM_SEQ_LEN]
        origin_batch_size = user_his.size(0)
        real_his_len = min(self.max_his_len, user_his_len[i % origin_batch_size].item())

        user_his_text = [self.item_text[user_his[i % origin_batch_size, user_his_len[i % origin_batch_size].item() - real_his_len + j].item()] \
                for j in range(real_his_len)]
        candidate_text = [self.item_text[idxs[i,j]]
                for j in range(idxs.shape[1])]
        candidate_text_order = [str(j) + '. ' + self.item_text[idxs[i,j].item()]
                for j in range(idxs.shape[1])]
        candidate_idx = idxs[i].tolist()
        ground_truth_text = self.item_text[interaction['item_id'][i]]

        return user_his_text, candidate_text, candidate_text_order, candidate_idx, ground_truth_text
    
    def get_batch_examples(self, total_examples, idx):
        examples = total_examples[idx]
        examples_text = []

        for i in range(1):
            user_id, history_seq, seq_len, target_item, candidate_items = examples[i]
            user_his_text = [self.item_text[history_seq[j][0]] for j in range(seq_len)]
            target_position = np.random.randint(0, self.recall_budget)
            candidate_items_with_target = candidate_items[:target_position] + [target_item] + candidate_items[target_position:]
            candidate_text = [self.item_text[candidate_items_with_target[j]] for j in range(len(candidate_items_with_target))]
            candidate_text_order = [str(j) + '. ' + self.item_text[candidate_items_with_target[j]] for j in range(len(candidate_items_with_target))]
            candidate_idx = candidate_items_with_target
            ground_truth_text = self.item_text[target_item]
            target = candidate_text_order[target_position]
            pseudo_answer =candidate_text_order[:target_position]+candidate_text_order[target_position+1:]
            np.random.shuffle(pseudo_answer)
            pseudo_answer = [target] + pseudo_answer
            pseudo_answer = '\n'.join(pseudo_answer)
            examples_text.append({'user_his_text': user_his_text, 'candidate_text': candidate_text, 'candidate_text_order': candidate_text_order, 'candidate_idx': candidate_idx, 'ground_truth_text': ground_truth_text, 'pseudo_answer': pseudo_answer})
        return examples_text
    

    def construct_prompt(self, dataset_name, user_his_text, candidate_text_order):
        if dataset_name == 'ml-1m':
            prompt = f"I've watched the following movies in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. \n" \
                    f"Please show me your rank results. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list. Do not give any explanations"
        elif dataset_name == 'Games' or dataset_name == 'Luxury':
            products = dataset_name
            if dataset_name == 'Luxury':
                products = 'Luxury Beauty'
            prompt = f"I've purchased the following {products} in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate {products} that I can consider to purchase next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} {products} by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Do not give any explanations\n" \
                    f"Please show me your rank results. Split your output with line break. You can NOT generate {products} that are not in the given candidate list. You MUST rank the given candidate {products}."
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt
    
    def construct_prompt_COT_Ori(self, dataset_name, user_his_text, candidate_text_order):
        if dataset_name == 'ml-1m':
            prompt = f"I've watched the following movies in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step\n" \
                    f"Please show me your rank results. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list. Do not give any explanations"
        elif dataset_name == 'Games' or dataset_name == 'Luxury':
            
            products = dataset_name
            if dataset_name == 'Luxury':
                products = 'Luxury Beauty'
                
            prompt = f"I've purchased the following {products} in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate {products} that I can consider to purchase next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} {products} by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step Do not give any explanations\n" \
                    f"Please show me your rank results. Split your output with line break. You can NOT generate {products} that are not in the given candidate list. You MUST rank the given candidate {products}."
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt
    
    def construct_prompt_ICL_pre(self, dataset_name, user_his_text, candidate_text_order):
        recent_item = user_his_text[-1][user_his_text[-1].find('. ') + 2:]
        if dataset_name == 'ml-1m':
            prompt = f"I've watched the following movies in the pastn order:\n{user_his_text[:-1]}\n\n" \
                    f"Then if I ask you to recommend a new movie to me according to my watching history, you should recommend {recent_item} and " \
                    f"now that I've just watched {recent_item}, " \
                    f"there are {self.recall_budget} candidate movies that I can watch next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history.\n" \
                    f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
        elif dataset_name == 'Games' or dataset_name == 'Books' or dataset_name == 'Magazine' or dataset_name == 'Luxury':
            products = dataset_name
            if dataset_name == 'Luxury':
                products = 'Luxury Beauty'
            prompt = f"I've purchased the following {products} in the past in order:\n{user_his_text[:-1]}\n\n" \
                    f"Then if I ask you to recommend a new product to me according to the given purchasing history, you should recommend {recent_item} and " \
                    f"now that I've just purchased {recent_item}, " \
                    f"there are {self.recall_budget} candidate {products} that I can consider to purchase next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} {products} by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. \n" \
                    f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate {products}. You can not generate {products} that are not in the given candidate list."
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt
    
    def construct_prompt_ICL(self, dataset_name, user_his_text, candidate_text_order, examples_text):
        if dataset_name == 'ml-1m':
            prompt = ''
            for example in examples_text:
                temp = f"Question: Given the historical interactions: \n{example['user_his_text']}\n\n" \
                    f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{example['candidate_text_order']}\n" \
                    f"What is the order of {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. \n" \
                    f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list.\n"\
                    f"Answer: {example['pseudo_answer']}\n\n"
                prompt += temp
            prompt += f"Question: Given the historical interactions:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. \n" \
                    f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."\
                    f"Answer:"
        
        
        elif dataset_name == 'Games' or dataset_name == 'Books' or dataset_name == 'Magazine' or dataset_name == 'Luxury':
            products = dataset_name
            if dataset_name == 'Luxury':
                products = 'Luxury Beauty'
                
            prompt = ''
            for example in examples_text:
                temp = f"Question: Given the historical interactions: \n{example['user_his_text']}\n\n" \
                    f"Now there are {self.recall_budget} candidate {products} that I can purchase next:\n{example['candidate_text_order']}\n" \
                    f"What is the order of {self.recall_budget} {products} by measuring the possibilities that I would like to purchase next most, according to my purchase history. \n" \
                    f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate {products}. You can not generate {products} that are not in the given candidate list.\n"\
                    f"Answer: {example['pseudo_answer']}\n\n"
                prompt += temp
            prompt += f"Question: Given the historical interactions:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate {products} that I can purchase next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} {products} by measuring the possibilities that I would like to purchase next most, according to my purchase history. \n" \
                    f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate {products}. You can not generate {products} that are not in the given candidate list."\
                    f"Answer:"
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt
    

    def inference_COT(self, dataset_name, user_his_text, candidate_text_order):
        if dataset_name == 'ml-1m':
            
            system_message = f"You are a professional recommender assistant, your task is answer user's questions regarding to the movies. \n "
            message_content = [{'role': "system", "content": system_message}]
            
            conv1 = f"""
            I have watched the following movies in the past in order: {user_his_text[:-1]},\n Could you please analysis the user preference according to the commonality of the movies' category, actor, mood?
            """
            
            message_content.append({
                'role': "user",
                "content": conv1
            })
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            conv4 = f"""
            Based on the analyze of user preference, Given the historical interaction:\n{user_his_text}\n There are {self.recall_budget} candidate movies that I can watch next: {candidate_text_order},\n Please rank these movies, put the movies of high probabilities first, and split your output with line break. You MUST rank the given candidates. You cannot generate movies that are not in the given candidate list.
            """
            
            message_content.append({
                'role': "user",
                "content": conv4
            })
            
            results ={}
            results['rank']= self.client.chat.completions.create(
                model = self.model_name,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': results['rank']
            })

            return results, message_content
                    
        elif dataset_name == 'Games':
            
            system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the products. \n "
            message_content = [{'role': "system", "content": system_message}]
            
            conv1 = f"""
            I have purchased the following products in the past in order: {user_his_text[:-1]},\n Could you please analysis the user preference according to the commonality of the game's category, publisher reputation, review and rating, shopping service?
            """
            
            message_content.append({
                'role': "user",
                "content": conv1
            })
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            conv4 = f"""
            Based on the analyze of user preference, Given the historical interaction:\n{user_his_text}\n There are {self.recall_budget} candidate products that I can purchase next: {candidate_text_order},\n Please rank these products, put the products of high probabilities first, and split your output with line break. You MUST rank the given candidates. You cannot generate products that are not in the given candidate list.
            """
            message_content.append({
                'role': "user",
                "content": conv4
            })
            
            results ={}
            results['rank']= self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': results['rank']
            })

            return results, message_content            
        
        elif dataset_name == 'Luxury':
            products = 'Luxury Beauty'
            system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the {products}. \n "
            
            message_content = [{'role': "system", "content": system_message}]
            
            conv1 = f"""
            I have purchased the following {products} in the past in order: {user_his_text[:-1]},\n Could you please analysis the user preference according to the commonality of the {products} category, seller reputation, price, purpose?
            """
            
            message_content.append({
                'role': "user",
                "content": conv1
            })
            response = self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            conv4 = f"""
            Based on the analyze of user preference, Given the historical interaction:\n{user_his_text}\n There are {self.recall_budget} candidate {products} that I can purchase next: {candidate_text_order},\n Please rank these {products}, put the {products} of high probabilities first, and split your output with line break. You MUST rank the given candidates. You cannot generate {products} that are not in the given candidate list.
            """
            message_content.append({
                'role': "user",
                "content": conv4
            })
            
            results ={}
            results['rank']= self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': results['rank']
            })

            return results, message_content
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')


    def inference_COT_example(self, dataset_name, user_his_text, candidate_text_order, example):
        if dataset_name == 'ml-1m':
            
            system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the movies. \n Here is some other users example: Question: Given the historical interactions: \n{example['user_his_text']}\n\n" \
                    f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{example['candidate_text_order']}\n" \
                    f"What is the order of {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history.\n" \
                    f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list.\n"\
                    f"Rank: {example['pseudo_answer']}\n\n"
            
            
            message_content = [{'role': "system", "content": system_message}]
            
            conv1 = f"""
            I have watched the following movies in the past in order: {user_his_text[:-1]},\n Could you please analysis the user preference according to the commonality of the movies' category, actor, mood?
            """
            
            message_content.append({
                'role': "user",
                "content": conv1
            })
            response = self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            conv4 = f"""
            Based on the analyze of user preference, Given the historical interaction:\n{user_his_text}\n There are {self.recall_budget} candidate movies that I can watch next: {candidate_text_order},\n Please rank these movies, put the movies of high probabilities first, and split your output with line break. You MUST rank the given candidates. You cannot generate movies that are not in the given candidate list.
            """
            message_content.append({
                'role': "user",
                "content": conv4
            })
            
            results ={}
            results['rank']= self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': results['rank']
            })
            return results, message_content

        elif dataset_name == 'Games':
            
            system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the games. \n Here is some other users example: Question: Given the historical interactions: \n{example['user_his_text']}\n\n" \
                    f"Now there are {self.recall_budget} candidate Games that I can purchase next:\n{example['candidate_text_order']}\n" \
                    f"What is the order of {self.recall_budget} games by measuring the possibilities that I would like to purchase next most, according to my purchase history. \n" \
                    f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate products. You can not generate games that are not in the given candidate list.\n"\
                    f"Rank: {example['pseudo_answer']}\n\n"
            
            
            message_content = [{'role': "system", "content": system_message}]
            
            conv1 = f"""
            I have purchased the following products in the past in order: {user_his_text[:-1]},\n Could you please analysis the user preference according to the commonality of the game's category, publisher reputation, review and rating, shopping service?
            """
            
            message_content.append({
                'role': "user",
                "content": conv1
            })
            response = self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            conv4 = f"""
            Based on the analyze of user preference, Given the historical interaction:\n{user_his_text}\n There are {self.recall_budget} candidate products that I can purchase next: {candidate_text_order},\n Please rank these products, put the products of high probabilities first, and split your output with line break. You MUST rank the given candidates. You cannot generate products that are not in the given candidate list.
            """
            message_content.append({
                'role': "user",
                "content": conv4
            })
            
            results ={}
            results['rank']= self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': results['rank']
            })
            
            return results, message_content


        elif dataset_name == 'Luxury':
            products = 'Luxury Beauty'
            system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the {products}. \n Here is some other users example: Question: Given the historical interactions: \n{example['user_his_text']}\n\n" \
                    f"Now there are {self.recall_budget} candidate {products} that I can purchase next:\n{example['candidate_text_order']}\n" \
                    f"What is the order of {self.recall_budget} {products} by measuring the possibilities that I would like to purchase next most, according to my purchase history. \n" \
                    f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate products. You can not generate Magazine that are not in the given candidate list.\n"\
                    f"Rank: {example['pseudo_answer']}\n\n"
            
            message_content = [{'role': "system", "content": system_message}]
            
            conv1 = f"""
            I have purchased the following {products} in the past in order: {user_his_text[:-1]},\n Could you please analysis the user preference according to the commonality of the {products} category, seller reputation, price, purpose?
            """
            
            message_content.append({
                'role': "user",
                "content": conv1
            })
            response = self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            conv4 = f"""
            Based on the analyze of user preference, Given the historical interaction:\n{user_his_text}\n There are {self.recall_budget} candidate {products} that I can purchase next: {candidate_text_order},\n Please rank these {products}, put the {products} of high probabilities first, and split your output with line break. You MUST rank the given candidates. You cannot generate {products} that are not in the given candidate list.
            """
            message_content.append({
                'role': "user",
                "content": conv4
            })
            
            results ={}
            results['rank']= self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': results['rank']
            })
            return results, message_content
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        
        
    def inference_COT_reflex(self, dataset_name, user_his_text, candidate_text_order, example):
        if dataset_name == 'ml-1m':
            
            if example == None:
                system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the movies. \n \n"
            else:  
                system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the movies. \n Here is some other users example: Question: Given the historical interactions: \n{example['user_his_text']}\n\n" \
                        f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{example['candidate_text_order']}\n" \
                        f"What is the order of {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. \n" \
                        f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list.\n"\
                        f"Rank: {example['pseudo_answer']}\n\n"
            
            
            message_content = [{'role': "system", "content": system_message}]
            
            conv1 = f"""
            I have watched the following movies in the past in order: {user_his_text[:-1]},\n Could you please analysis the user preference according to the commonality of the movies' category, actor, mood?
            """
            
            message_content.append({
                'role': "user",
                "content": conv1
            })
            response = self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name_long,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            conv2 = f"""
            What is the next possible movie I would like to watch next? {user_his_text[-1]} or {np.random.choice(self.item_text,1)[0]}\n
            """

            message_content.append({
                'role': "user",
                "content": conv2
            })
            response = self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name_long,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            conv3 = f"""
            The answer is {user_his_text[-1]}, is that consist with the previous preferences? From what aspect will you recommend this movie to user? Update your preference analysis on this user.
            """
            message_content.append({
                'role': "user",
                "content": conv3
            })
            response = self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name_long,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            conv4 = f"""
            Based on the updated user preference, I have the following candidate movies that I can watch next: {candidate_text_order},\n Please rank these movies, put the movies of high probabilities first, and split your output with line break. You MUST rank the given candidates. You cannot generate movies that are not in the given candidate list.
            """
            message_content.append({
                'role': "user",
                "content": conv4
            })
            
            results ={}
            results['rank']= self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo',
                model = self.model_name_long,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': results['rank']
            })
            return results, message_content
        
        
        elif dataset_name == 'Games' or dataset_name == 'Books' or dataset_name == 'Magazine' or dataset_name == 'Luxury':
            products = dataset_name
            if dataset_name == 'Luxury':
                products = 'Luxury Beauty'
                
            if example == None:

                system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the {products}. \n \n" 
            else:
                            
                system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the {products}. \n Here is some other users example: Question: Given the historical interactions: \n{example['user_his_text']}\n\n" \
                        f"Now there are {self.recall_budget} candidate {products} that I can purchase next:\n{example['candidate_text_order']}\n" \
                        f"What is the order of {self.recall_budget} {products} by measuring the possibilities that I would like to purchase next most, according to my purchase history.\n" \
                        f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate {products}. You can not generate {products} that are not in the given candidate list.\n"\
                        f"Rank: {example['pseudo_answer']}\n\n"
                
            
            
            message_content = [{'role': "system", "content": system_message}]
            
            if dataset_name == 'Games':
                conv1 = f"""
                I have purchased the following games in the past in order: {user_his_text[:-1]},\n Could you please analysis the user preference according to the commonality of the game's category, publisher reputation, review and rating, shopping service?
                """
            elif dataset_name == 'Books':
                
                conv1 = f"""
                I have purchased the following Books in the past in order: {user_his_text[:-1]},\n Could you please analysis the user preference according to the commonality of the {products} category, author style, review and rating, price, publish date?
                """
            elif dataset_name == 'Magazine':
                    
                    conv1 = f"""
                    I have purchased the following Magazine in the past in order: {user_his_text[:-1]},\n Could you please analysis the user preference according to the commonality of the {products} category, review and rating, price, publish date?
                    """
            elif dataset_name == 'Luxury':
                        
                conv1 = f"""
                I have purchased the following {products} in the past in order: {user_his_text[:-1]},\n Could you please analysis the user preference according to the commonality of the {products} category, seller reputation, price, purpose?
                """
                        
            message_content.append({
                'role': "user",
                "content": conv1
            })
            response = self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo-1106',
                model = self.model_name_long,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            conv2 = f"""
            What is the next possible {products} I would like to purchase next? {user_his_text[-1]} or {np.random.choice(self.item_text,1)[0]}\n
            """

            message_content.append({
                'role': "user",
                "content": conv2
            })
            response = self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo-1106',
                model = self.model_name_long,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            conv3 = f"""
            The answer is {user_his_text[-1]}, are you correct? is that consist with the previous preferences? From what aspect will you recommend this {products} to user? Update your preference analysis on this user.
            """
            message_content.append({
                'role': "user",
                "content": conv3
            })
            response = self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo-1106',
                model = self.model_name_long,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            conv4 = f"""
            Based on the updated user preference, I have the following candidate {products} that I can purchase next: {candidate_text_order},\n Please rank these {products}, put the {products} of high probabilities first, and split your output with line break. You MUST rank the given candidates. You cannot generate {products} that are not in the given candidate list.
            """
            message_content.append({
                'role': "user",
                "content": conv4
            })
            
            results ={}
            results['rank']= self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo-1106',
                model = self.model_name_long,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': results['rank']
            })
            
            return results, message_content
        
                
    
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt
    
    def inference_COT_reflex_long(self, dataset_name, user_his_text, candidate_text_order, example):
        if dataset_name == 'ml-1m':
            if example == None:
                system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the movies. \n \n" 
            else:
            
                system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the movies. \n Here is some other users example: Question: Given the historical interactions: \n{example['user_his_text']}\n\n" \
                        f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{example['candidate_text_order']}\n" \
                        f"What is the order of {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. \n" \
                        f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list.\n"\
                        f"Rank: {example['pseudo_answer']}\n\n"
            
            
            message_content = [{'role': "system", "content": system_message}]
            conv1 = f"""
                I have watched the following movies in the past in order: {user_his_text[:-self.reflex_step]},\n Could you please analysis the user preference according to the commonality of the movies' category, actor, mood?
                """
                
            message_content.append({
                'role': "user",
                "content": conv1
            })
            response = self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo-1106',
                model = self.model_name_long,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            for k in range(self.reflex_step):

                
                conv2 = f"""
                What is the next possible movie I would like to watch next? {user_his_text[-self.reflex_step+k]} or {np.random.choice(self.item_text,1)[0]}\n
                """

                message_content.append({
                    'role': "user",
                    "content": conv2
                })
                response = self.client.chat.completions.create(
                    # model = 'gpt-3.5-turbo-1106',
                    model = self.model_name_long,
                    messages = message_content
                ).choices[0].message.content
                
                message_content.append({
                    'role': "assistant",
                    'content': response
                })
                
                conv3 = f"""
                The answer is {user_his_text[-self.reflex_step+k]}, is that consist with the previous preferences? From what aspect will you recommend this movie to user? Update your preference analysis on this user.
                """
                message_content.append({
                    'role': "user",
                    "content": conv3
                })
                response = self.client.chat.completions.create(
                    # model = 'gpt-3.5-turbo-1106',
                    model = self.model_name_long,
                    messages = message_content
                ).choices[0].message.content
                
                message_content.append({
                    'role': "assistant",
                    'content': response
                })
            
            conv4 = f"""
            Based on the updated user preference, I have the following candidate movies that I can watch next: {candidate_text_order},\n Please rank these movies, put the movies of high probabilities first, and split your output with line break. You MUST rank the given candidates. You cannot generate movies that are not in the given candidate list.
            """
            message_content.append({
                'role': "user",
                "content": conv4
            })
            
            results ={}
            results['rank']= self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo-1106',
                model = self.model_name_long,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': results['rank']
            })
            
            return results, message_content
        
                
        elif dataset_name == 'Games' or dataset_name == 'Books' or dataset_name == 'Magazine' or dataset_name == 'Luxury':
            products = dataset_name
            if dataset_name == 'Luxury':
                products = 'Luxury Beauty'
                
            if example == None:

                system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the {products}. \n\n" 
            else:
                system_message = f"You are a professional recommender system, your task is answer user's questions regarding to the {products}. \n Here is some other users example: Question: Given the historical interactions: \n{example['user_his_text']}\n\n" \
                f"Now there are {self.recall_budget} candidate {products} that I can purchase next:\n{example['candidate_text_order']}\n" \
                f"What is the order of {self.recall_budget} {products} by measuring the possibilities that I would like to purchase next most, according to my purchase history.\n" \
                f"Please show me your ranking results. Split your output with line break. You MUST rank the given candidate {products}. You can not generate {products} that are not in the given candidate list.\n"\
                f"Rank: {example['pseudo_answer']}\n\n"
                
            
            
            message_content = [{'role': "system", "content": system_message}]
            if dataset_name == 'Games':
                conv1 = f"""
                    I have purchased the following games in the past in order: {user_his_text[:-self.reflex_step]},\n Could you please analysis the user preference according to the commonality of the game's category, publisher reputation, review and rating, shopping service?
                    """
            elif dataset_name == 'Books':
                
                conv1 = f"""
                I have purchased the following Books in the past in order: {user_his_text[:-self.reflex_step]},\n Could you please analysis the user preference according to the commonality of the {products} category, author style, review and rating, price, publish date?
                """
            elif dataset_name == 'Magazine':
                        
                conv1 = f"""
                I have purchased the following Magazine in the past in order: {user_his_text[:-self.reflex_step]},\n Could you please analysis the user preference according to the commonality of the {products} category, review and rating, price, publish date?
                """
            elif dataset_name == 'Luxury':
                            
                conv1 = f"""
                I have purchased the following {products} in the past in order: {user_his_text[:-self.reflex_step]},\n Could you please analysis the user preference according to the commonality of the {products} category, seller reputation, price, purpose?
                """
                    
            message_content.append({
                'role': "user",
                "content": conv1
            })
            response = self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo-1106',
                model = self.model_name_long,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': response
            })
            
            
            for k in range(self.reflex_step):

                
                conv2 = f"""
                What is the next possible {products} I would like to purchase next? {user_his_text[-self.reflex_step+k]} or {np.random.choice(self.item_text,1)[0]}\n
                """

                message_content.append({
                    'role': "user",
                    "content": conv2
                })
                response = self.client.chat.completions.create(
                    # model = 'gpt-3.5-turbo-1106',
                    model = self.model_name_long,
                    messages = message_content
                ).choices[0].message.content
                
                message_content.append({
                    'role': "assistant",
                    'content': response
                })
            
                
                conv3 = f"""
                The answer is {user_his_text[-self.reflex_step+k]}, are you correct? is that consist with the previous preferences? From what aspect will you recommend this {products} to user? Update your preference analysis on this user.
                """
                message_content.append({
                    'role': "user",
                    "content": conv3
                })
                response = self.client.chat.completions.create(
                    # model = 'gpt-3.5-turbo-1106',
                    model = self.model_name_long,
                    messages = message_content
                ).choices[0].message.content
                
                message_content.append({
                    'role': "assistant",
                    'content': response
                })
            
            
            conv4 = f"""
            Based on the updated user preference, I have the following candidate {products} that I can purchase next: {candidate_text_order},\n Please rank these {products}, put the {products} of high probabilities first, and split your output with line break. You MUST rank the given candidates. You cannot generate {products} that are not in the given candidate list.
            """
            message_content.append({
                'role': "user",
                "content": conv4
            })
            
            results ={}
            results['rank']= self.client.chat.completions.create(
                # model = 'gpt-3.5-turbo-1106',
                model = self.model_name_long,
                messages = message_content
            ).choices[0].message.content
            
            message_content.append({
                'role': "assistant",
                'content': results['rank']
            })
            
            return results, message_content
        
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt   

    def parsing_output_text(self, scores, batch_i, output):
        rec_items = output.split('\n')
        reced_list = []
        for i in range(len(rec_items)):
            item = rec_items[i]
            for j, item_title in enumerate(self.item_text):
                if item_title in item and item_title not in reced_list:
                    scores[batch_i][j] = len(rec_items)-i
                    reced_list.append(item_title)
                    break
        