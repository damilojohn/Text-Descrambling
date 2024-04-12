from modal import Secret, Stub, Image, Volume
from modal import enter, method
from pathlib import Path

VOL_MOUNT_PATH = Path('/vol')
BASE_MODEL = 'openai-community/gpt2-medium'

image = (
    Image.debian_slim(python_version='3.10')
    .pip_install(
        'transformers[torch]',
        'huggingface_hub'
    )
)

stub = Stub(image=image)
volume = Volume.from_name('finetune-text-descrambler')


@stub.cls(
    secrets=[Secret.from_name('hf-secret')],
    gpu=None,
    volumes={VOL_MOUNT_PATH: volume}

)
class Push_to_hub:
    @enter()
    def load_model(self):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import os
        # hf_token = os.environ['HF_TOKEN']

        self.model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=VOL_MOUNT_PATH/'model/',
            cache_dir=VOL_MOUNT_PATH / "model/"
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            BASE_MODEL,
            cache_dir=VOL_MOUNT_PATH/'tokenizer'
        )

    @method()
    def push_to_hub(self):
        self.model.push_to_hub('damilojohn/text-descrambling-gpt2')
        print('model successfully pushed to huggingface')


@stub.local_entrypoint()
def main():
    hub = Push_to_hub()
    hub.push_to_hub.remote()
    print('done......')



