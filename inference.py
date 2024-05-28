from modal import (Volume,
                   Stub,
                   Image,
                   enter,
                   method,
                   Secret)
from pathlib import Path


gpt2_image = (
    Image.debian_slim(python_version='3.10')
    .pip_install(
        'transformers[torch]',
        'accelerate',
        "datasets",
        'wandb',
        'huggingface_hub'
    )
)

VOL_MOUNT_PATH = Path("/vol")
BASE_MODEL = 'openai-community/gpt2-medium'

stub = Stub('serving-gpt2', image=gpt2_image)
volume = Volume.from_name('finetune-text-descrambler')


@stub.cls(
    volumes={VOL_MOUNT_PATH: volume},
    secrets=[Secret.from_name('hf-secret')],
    gpu='A10G')
class Descrambler:
    @enter()
    def load_model(self):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            BASE_MODEL,
            cache_dir=VOL_MOUNT_PATH / "tokenizer/"
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=VOL_MOUNT_PATH/'model/',
            cache_dir=VOL_MOUNT_PATH / "model/"
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @method()
    def generate(self, input):
        self.model.to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        input = [f"wrong sentence: {sentence} correct sentence:"
                 for sentence in input]
        input_tokens = self.tokenizer(input, return_tensors='pt',
                                      padding='max_length', truncation=True,
                                      max_length=128).to(self.device)       
        out = self.model.generate(**input_tokens,
                                  max_new_tokens=128,
                                  do_sample=True,
                                  num_beams=10
                                )
        for i, out in enumerate(out):
            decoded_output = self.tokenizer.decode(out,
                                                   skip_special_tokens=True)
            print(f"Input: {input[i]}")
            print(f"Output: {decoded_output}")


@stub.local_entrypoint()
def main():
    sentences = [
        # 'the which wiring flow. propose to diagram, method network a reflects signal We visualize',
        # 'the interaction networks. the gap Finally, analyze chemical the junction between synapse and we',
        # 'the process The pseudorandom number illustrated in is Mathematica. generator using',
        # 'in the of structure resulted decrease mutual signal in information. Introducing correlations input-output',
        # 'statistical estimators functionals. of various of consistent We investigate existence the bounded-memory',
        # 'rather negative sense. the question This in strong a is in resolved'
        ('method Therefore, we in propose that their parameters.' 
         'environments to simulate causal a relationships offer'),
        'image a we In paper, scene. nighttime single address in haze problem the removal this',
        'in noticeable are of different night light and sources These glow scenes. shapes introduce',
        'of proposed to is the function a function In combine loss loss new classification. addition,',
        'learned similarity functional of The gene pathways. perturbing the embeddings capture SGAs common',
        'main contribution of methods. the a is deficiencies function, established overcomes The which novel enhancement',
        'for ubiquitous WiFi The and backscatter systems. communications IoT ultra-low power offer connections',
        'The was programming programming constraint using then model developed language. python implemented',
        'the and interest gaining learning more Multimodal within more representation community. deep learning',
        'equalizers. a analog-to-digital paper frontend presents This for discrete-time an based analog (ADC)',
        'the and interest gaining learning more Multimodal within more representation community. deep learning ',
        'differential low-power The comparators. fully uses ADC clocked',

        ]
    model = Descrambler()
    response = model.generate.remote(sentences)
    print(response)
