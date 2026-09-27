import torch
from transformers import AutoTokenizer, AutoModel


# Pre-trained Transformer model
MODEL_NAME = "distilbert-base-uncased"


class V2VTransformer:

    def __init__(self):
        print("Loading Transformer model...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)

        # We are using the Transformer as a feature extractor
        self.model.eval()

        print("Transformer loaded successfully!")

    def encode_messages(self, messages):

        # Convert text into Transformer tokens
        inputs = self.tokenizer(
            messages,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Generate Transformer representations
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Take the representation of the first token [CLS]
        embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings


if __name__ == "__main__":

    transformer = V2VTransformer()

    test_messages = [
        "Vehicle ahead stopped suddenly 20 m ahead.",
        "Traffic is slowing near the junction.",
        "Road construction 5 km ahead."
    ]

    embeddings = transformer.encode_messages(test_messages)

    print("\nTransformer test successful!")
    print("Number of messages:", len(test_messages))
    print("Embedding shape:", embeddings.shape)