import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.utils import init_seed, init_logger, set_color
from utils import get_model
from trainer import SelectedUserTrainerICL


def evaluate(model_name, model_path, dataset_name, pretrained_file, model_version, inference_model, **kwargs):

    props = [f'props/{dataset_name}.yaml', 'props/overall.yaml']
    print(props)
    print(f"model_name: {model_name}, model_version: {model_version}, inference_model: {inference_model}")
    model_class = get_model(model_name)

    # configurations initialization
    config = Config(model=model_class, dataset=dataset_name, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    config['model_version'] = model_version
    config['inference_model'] = inference_model
    config['model_path'] = model_path
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)


    dataset = SequentialDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization

    trainer = SelectedUserTrainerICL(config, model, dataset)


    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])
    print(test_result)
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    output_res = []
    for u, v in test_result.items():
        output_res.append(f'{v}')
    logger.info('\t'.join(output_res))

    return config['model'], config['dataset'], {
        'valid_score_bigger': config['valid_metric_bigger'],
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str, default="LLM", help="model name")
    parser.add_argument('--d', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('--p', type=str, default='', help='pre-trained model path')
    parser.add_argument('--model_version', default = 'COT')
    parser.add_argument('--inference_model', default='vicuna-13b', help='model name while setup fastchat inference model')
    parser.add_argument('--reflex_step', default=2, type=int, help='reflex step for icl')
    args, unparsed = parser.parse_known_args()
    print(f"args: {args}")

    # model_path in huggingface
    if args.inference_model in ['vicuna-13b-16k']:
        args.model_path = 'lmsys/vicuna-13b-v1.5-16k'
    elif args.inference_model in ['vicuna-13b']:
        args.model_path = 'lmsys/vicuna-13b-v1.5'
    elif args.inference_model in ['vicuna-7b-16k']:
        args.model_path = 'lmsys/vicuna-7b-v1.5-16k'
    elif args.inference_model in ["openchat-3.5"]: # supported by fastchat
        args.model_path = 'openchat/openchat_3.5'
    elif args.inference_model in ['yarn-mistral']:
        args.model_path = 'NousResearch/Yarn-Mistral-7b-128k'
    elif args.inference_model in ['longchat-7b']:
        args.model_path = 'lmsys/longchat-7b-16k'
    elif args.inference_model in ['gpt-3.5-turbo', 'gpt-4']:
        args.model_path = ''
        print('no need for model path')
    else:
        assert False, 'inference model not supported'

    evaluate(args.m, args.model_path, args.d, pretrained_file=args.p, model_version=args.model_version, inference_model=args.inference_model, reflex_step=args.reflex_step)
    
