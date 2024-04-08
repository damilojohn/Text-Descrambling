import modal
from modal import Stub, Image, Secret, Volume, enter, method
from pathlib import Path

gpt2_image = (
    Image.debian_slim()
    .pip_install(
        'accelerate==0.18.0',
        "datasets == 2.10.1",
        'transformers[torch]',
        'wandb'
    )
)
VOL_MOUNT_PATH = Path("/vol")
BASE_MODEL = 'openai-community/gpt2-medium'

stub = Stub(name='finetune_text_descrambler-gpt2', image=gpt2_image)
output_vol = Volume.from_name('finetune-text-descrambler', create_if_missing=True)

# handling preemption
restart_tracker_dict = modal.Dict.from_name(
    "finetune-text_descrambler", create_if_missing=True
)


def track_restarts(restart_tracker: modal.Dict) -> int:
    if not restart_tracker.contains("count"):
        preemption_count = 0
        print(f"Starting first time. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    else:
        preemption_count = restart_tracker.get("count") + 1
        print(f"Restarting after pre-emption. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    return preemption_count


def _prepare_train():
    import torch
    from transformers import (
        GPT2LMHeadModel,
        GPT2Tokenizer,
        TrainingArguments,
        Trainer
    )
    from datasets import load_dataset
    import random
    import numpy as np
    import wandb

    restarts = track_restarts(restart_tracker_dict)

    wandb.login()
    config = {
        'model name': 'GPT2-medium',
        'learning_rate': '3e-5',
        'architecture': 'decoder-only',
        'context_length': 'not set',
        'training_strategy':
        'sliding window with input tokens set to 101 during loss calculations'
        }
    wandb.init(project='Text Descrambling',
               entity='damilojohn',
               name='gpt2-medium-new strategy',
               config=config
               )
    seed_val = 12
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = 'cuda' if torch.cuda.is_avalilable() else 'cpu'

    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    model.to(device)

    print('the max model length for this token is {}'.format(tokenizer.model_max_length))
    print(
        f'''this model has 
        {sum([p.numel() for p in model.parameters()])//10**6:.1f}
        M parameters'''
        )
    # load the dataset
    dataset = load_dataset("damilojohn/Text-Descrambling")
    # Define a function to preprocess the data

    def preprocess_data(row):
        target_text = row['label']
        # add prompt to every row in the dataset
        input_text = f'''wrong sentence: {row['text']} correct sentence:'''
        # find the length of the input prompt 
        prompt_len = len(tokenizer(input_text).input_ids)
        input = tokenizer(f'{input_text} {target_text} <|endoftext|>',
                          padding='max_length', truncation=True, 
                          max_length=128, return_tensors='pt').to(device)
        input_ids, attention_mask = input.input_ids, input.attention_mask
        # turn all of the tokens before the actual correct sentence to -100
        # so loss is only calculated for generation after 'correct sentence:'
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        # Turn all pad tokens to -100
        labels[labels == tokenizer.pad_token_id] = -100
        assert (labels == -100).sum() > len(labels), "Labels are all -100,something is wrong."
        # if (labels == -100).sum() == len(labels) - 1:
        #         raise
        return {'input_ids': input_ids.squeeze(),
                'attention_mask': attention_mask.squeeze(), 
                'labels': labels.squeeze(),
                'prompt': input_text}
    processed_data = dataset.map(preprocess_data)
    processed_data.set_format(type='torch', columns=['input_ids',
                                                     'attention_mask', 'labels'
                                                     ])
    batch_size = 128
# Training arguments and Trainer
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        num_train_epochs=5,
        learning_rate=3e-5,
        output_dir=str(VOL_MOUNT_PATH / "model"),
        logging_dir=str(VOL_MOUNT_PATH / "logs"),
        logging_strategy='steps',
        logging_steps=100,
        load_best_model_at_end=True,
        save_strategy='steps',
        evaluation_strategy='steps',
        save_steps=100,
        save_total_limit=2,
        push_to_hub=False,
        report_to='wandb',
        fp16=True
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_data['train'],
        eval_dataset=processed_data['val'],
        )
    trainer.train()
    wandb.finish()
    try:
        resume = restarts > 0
        if resume:
            print("resuming from checkpoint")
        trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:  # handle possible preemption
        print("received interrupt; saving state and model")
        trainer.save_state()
        trainer.save_model()
        raise

    model.save_pretrained(str(VOL_MOUNT_PATH / "model"))
    tokenizer.save_pretrained(str(VOL_MOUNT_PATH / "tokenizer"))
    output_vol.commit()
    print("✅ done")


@stub.function(
    gpu='A10g',
    timeout=60*60*2,
    secret=Secret.from_name('wandb-api'),
    volumes={VOL_MOUNT_PATH: output_vol},
    _allow_background_volume_commits=True,)
def finetune():
    print('started training')
    _prepare_train()


@stub.cls(volumes={VOL_MOUNT_PATH: output_vol})
class Descrambler:
    @enter
    def load_model(self):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            BASE_MODEL,
            cache_dir=VOL_MOUNT_PATH / "tokenizer/"
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            BASE_MODEL,
            cache_dir=VOL_MOUNT_PATH / "model/"
        )

    @method
    def generate(self, input):
        input = [f"wrong sentence: {sentence} correct sentence:"
                 for sentence in input]
        input_tokens = self.tokenizer(input)
        out = self.model.generate(**input_tokens,
                                  max_new_tokens=60,
                                  do_sample=True,
                                  num_beams=5)
        out = self.tokenizer.decode(out[0])
        return (f'model input {input} model output:{out}')


@stub.local_entrypoint()
def main():
    sentences = [
        'the which wiring flow. propose to diagram, method network a reflects signal We visualize',
        'the interaction networks. the gap Finally, analyze chemical the junction between synapse and we',
        'the process The pseudorandom number illustrated in is Mathematica. generator using',
        'in the of structure resulted decrease mutual signal in information. Introducing correlations input-output',
        'statistical estimators functionals. of various of consistent We investigate existence the bounded-memory',
        'rather negative sense. the question This in strong a is in resolved']
    model = Descrambler()
    response = model.generate.remote(sentences)
    print(response)
