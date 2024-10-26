from transformers import AutoTokenizer, BertModel
import sys
import torch
import numpy as np

if __name__ == "__main__":
    # Load the pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    model = BertModel.from_pretrained("bert-base-uncased")
    
    # Parse command-line arguments for input sentences
    if len(sys.argv) != 3:
        print("Usage: python script.py <sentence1> <sentence2>")
        sys.exit(1)
    
    # Extract sentences from command-line arguments
    sentence1 = sys.argv[1]
    sentence2 = sys.argv[2]
    
    sentences = [sentence1, sentence2]
    
    # Tokenize each sentence
    sentences_tokenized = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    # Run sentences through the model
    with torch.no_grad():  # Disable gradients for inference
        output = model(**sentences_tokenized)
    
    # Create two vector representations for each sentence
    # pool_vectors (2x768): vector created by performing max-pooling over the hidden states
    pool_vectors = torch.max(output.last_hidden_state, dim=1).values  # Max pooling over hidden states
    
    # cls_vectors (2x768): vector created by taking the CLS token embedding
    cls_vectors = output.last_hidden_state[:, 0, :]  # CLS token is the first token

    # Compute cosine similarity between representations
    cosine_pooling = torch.nn.functional.cosine_similarity(pool_vectors[0].unsqueeze(0), pool_vectors[1].unsqueeze(0))
    cosine_cls = torch.nn.functional.cosine_similarity(cls_vectors[0].unsqueeze(0), cls_vectors[1].unsqueeze(0))
    
    # Print out the similarity scores rounded to 2 decimal places
    print(np.round(cosine_pooling.item(), 2), np.round(cosine_cls.item(), 2))