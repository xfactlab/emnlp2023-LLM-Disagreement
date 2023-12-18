import argparse

def get_args():
    """ args for """
    parser = argparse.ArgumentParser(description="Args for Generater")

    parser.add_argument('--data_dir',
                    default = "./data/inputs/dict_snli.json",
                    type = str,
    )
    parser.add_argument('--few_shot_data_dir',
                    default = "./data/inputs/dict_snli_5_few_shot.json",
                    type = str,
    )
    parser.add_argument('--data_type',
                    default = "snli",
                    choices = ['snli','mnli','alphanli','Pavlick','anli','qnli','disnli'],
                    type = str,
    )
    parser.add_argument('--model',
                    default = "flan-t5-xxl",
                    type = str,
    )
    parser.add_argument('--model_dir',
                    default = "", #"/projects/shared/Models/stable_vicuna_13b",
                    type = str,
    )
    parser.add_argument('--dump',
                    default = True,
                    type = bool,
    )    
    parser.add_argument('--file_name',
                    default = None,
                    type = str,
    )    
    parser.add_argument('--out_dir',
                    default = None,
                    type = str,
    )
    parser.add_argument('--num_iter',
                    default = 1,
                    type = int,
    )
    parser.add_argument('--max_length',
                    default = 1,
                    type = int,
    )
    parser.add_argument('--do_sample',
                    default = True,
                    type = bool,
    )
    parser.add_argument('--num_samples',
                    default = 1,
                    type = int,
    )
    parser.add_argument('--timeout',
                    default = 200,
                    type = int,
    )
    parser.add_argument('--device',
                    default = 'cuda',
                    type = str,
    )
    parser.add_argument('--gen_type',
                    default = 'mce',
                    choices = ['lpe', 'mce'],
                    type = str,
    )
    parser.add_argument('--prompt_type',
                    default = 'p2',
                    choices = ['p1', 'p2'],
                    type = str,
    )
    parser.add_argument('--multi_gpu',
                    default = True,
                    type = bool,
    )
    parser.add_argument('--shuffle',
                    default = True,
                    type = bool,
    )
    parser.add_argument('--time',
                    default = True,
                    type = bool,
    )
    parser.add_argument('--few_shot_k',
                    default = 0, # 0,1,3,5
                    type = str,
    )
    parser.add_argument('--prompt_variation',
                    default = False,
                    type = bool,
    )
    args = parser.parse_args()
    return args