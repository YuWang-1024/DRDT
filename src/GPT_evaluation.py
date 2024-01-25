import argparse
import json
from openai import OpenAI
from tqdm import tqdm
def evaluate(args, api_key):
    client = OpenAI(api_key=api_key)
    model_name = 'gpt-4'
    path = 'parsed_output/' + args.path + '.json'
    with open(path, 'r') as f:
        data = json.load(f)
    results = {}
    for k, v in tqdm(data.items()):
        message_content = [{'role':'system', 'content': 'Here is a conversation between an AI and a human. You are a professional evaluator that evaluate the recommendation capacities of AI in the conversation. '}]
        message_content.append({'role': 'user', 'content': '[Conversation]\n' + v + '\n[End of Conversation]\n'})
        message_content.append({'role': 'user', 'content': '[Instruct] Please evaluate the intermediate analysis of AI in the conversation from following aspects: 1. [Whether movies in history are classified into right aspects], 2. [Accuracy in User Preference Analysis in Multi-Round Attempt], 3. [Understanding of User History], 4. [Aspects Coverages of User Understanding], 5. [Adherence to User Query and Constraints], 6. [Whether Correctly Predicted the User next Watching activity], 7. [Whether Figure out a major aspect that conclude the final watching activity]. \n Please give rating scores from 0-1 for each aspects, and explain them. Give the output using line break for each aspects.  [End of Instruct]\n'})
        print("*********start**********")
        print(k)
        print(message_content)
        output = client.chat.completions.create(
            model=model_name,
            messages = message_content
        ).choices[0].message.content
        print('*******output********')
        print(output)
        print("*********end**********")
        results[k] = output
    with open('evaluation_results/' + args.path + '_results.json', 'w') as f:
        json.dump(results, f)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='gpt4_3', help='[gpt4_0, gpt4_1, gpt4_2, gpt4_3, openchat_1]' )
    args, unparsed = parser.parse_known_args()
    api_base= ""
    api_key= ""
    evaluate(args, api_key)