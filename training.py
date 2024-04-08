from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import torch
import random 
import numpy as np
import wandb
from modal import Stub, Image
import os
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

wandb.login(
    key= WANDB_API_KEY
)

config = {'model name':'GPT2-medium',
          'learning_rate':'5e-5',
          'architecture' : 'decoder-only',
          'context_length':'not set',
          'training_strategy':'sliding window with input tokens set to 101 during loss calculations'
}

run = wandb.init(project='Text Descrambling',
           entity='damilojohn',
           name = 'gpt2-medium-new strategy',
           config=config
)

data = run.use_artifact('word-deshuffling:latest')
directory = data.download()
train_path = '/content/artifacts/word-deshuffling:v0/train (8).csv'
test_path = '/content/artifacts/word-deshuffling:v0/test (9).csv'
val_path = '/content/artifacts/word-deshuffling:v0/val (1).csv'

dataset = load_dataset('csv',data_files = {'train':[train_path],
                                         'test':[test_path],
                                         'val':[val_path]}
                       )
seed_val = 12
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
batch_size = 64

image = Image()
stub = Stub()

device ='cuda' if torch.cuda.is_avalilable() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium',padding_side='right')
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

model.to(device)

print('the max model length for this token is {}'.format(tokenizer.model_max_length))
print(f'this model has {sum([p.numel() for p in model.parameters()])//10**6:.1f}M parameters')


@stub.function()
# Define a function to preprocess the data
def preprocess_data(row):
    target_text = row['label']
    #add prompt to every row in the dataset
    input_text = f'''wrong sentence: {row['text']} correct sentence:'''
    #find the length of the input prompt 
    prompt_len = len(tokenizer(input_text).input_ids)
    input = tokenizer(f'{input_text} {target_text} <|endoftext|>', padding='max_length', truncation=True, max_length=128, return_tensors ='pt').to(device)
    input_ids,attention_mask = input.input_ids,input.attention_mask
    #turn all of the tokens before the actual correct sentence to -100 so loss is only calculated for generation after 'correct sentence:'
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    # Turn all pad tokens to -100
    labels[labels == tokenizer.pad_token_id] = -100
    assert (labels == -100).sum() > len(labels), f"Labels are all -100, something wrong."
    # if (labels == -100).sum() == len(labels) - 1:
    #         raise
    return {'input_ids': input_ids.squeeze(),
            'attention_mask':attention_mask.squeeze(),'labels': labels.squeeze(),
            'prompt':input_text}


processed_data = dataset.map(preprocess_data)
processed_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
# Training arguments and Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    num_train_epochs=5,
    learning_rate=3e-5,
    output_dir='./results',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='wandb',
    fp16=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_data['train'],
    eval_dataset = processed_data['val'],
)
trainer.train()
