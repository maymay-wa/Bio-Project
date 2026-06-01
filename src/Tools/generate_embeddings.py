from transformers import BertModel, BertTokenizer
import torch
import re
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 1. Load ProtBERT (Run this once, it will download the model weights)
print("Loading ProtBERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

def get_protbert_embedding(sequence):
    """Generate ProtBERT embedding for a protein sequence."""
    # 2. Format: Add spaces and handle rare/unknown amino acids
    spaced_seq = " ".join(list(sequence))
    spaced_seq = re.sub(r"[UZOB]", "X", spaced_seq)
    
    # 3. Tokenize the sequence
    encoded_input = tokenizer(spaced_seq, return_tensors='pt')
    
    # 4. Pass through the model
    with torch.no_grad():
        output = model(**encoded_input)
        
    # 5. Extract the [CLS] token embedding (index 0)
    # This grabs the final 1024-dimensional vector representing the whole protein
    cls_embedding = output.last_hidden_state[0, 0, :].numpy()
    return cls_embedding


def generate_embeddings_csv(input_file, output_file, label_file=None):
    """
    Generate embeddings for sequences in input_file and save to CSV.
    
    Args:
        input_file: Path to file containing protein sequences (one per line)
        output_file: Path to output CSV file
        label_file: Optional path to file with labels (one per line, corresponding to sequences)
    """
    
    # Read sequences
    print(f"Reading sequences from {input_file}...")
    with open(input_file, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    
    # Read labels if provided
    labels = None
    if label_file:
        print(f"Reading labels from {label_file}...")
        with open(label_file, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        if len(labels) != len(sequences):
            print(f"Warning: Number of labels ({len(labels)}) doesn't match sequences ({len(sequences)})")
    
    print(f"Generating embeddings for {len(sequences)} sequences...")
    
    # Generate embeddings
    embeddings = []
    for i, seq in enumerate(sequences):
        if (i + 1) % 10 == 0:
            print(f"  Processing sequence {i + 1}/{len(sequences)}")
        
        try:
            embedding = get_protbert_embedding(seq)
            embeddings.append(embedding)
        except Exception as e:
            print(f"  Error processing sequence {i}: {e}")
            embeddings.append(np.full(1024, np.nan))
    
    # Create DataFrame
    embedding_columns = [f"dim_{j}" for j in range(1024)]
    df = pd.DataFrame(embeddings, columns=embedding_columns)
    
    # Add sequence column
    df.insert(0, 'sequence', sequences)
    
    # Add label column if provided
    if labels:
        df.insert(1, 'label', labels)
    
    # Add ID column
    df.insert(0, 'id', range(len(sequences)))
    
    # Save to CSV
    print(f"Saving embeddings to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Done! Saved {len(sequences)} embeddings to {output_file}")
    
    return df


if __name__ == "__main__":
    # Example usage - modify these paths as needed
    
    # For training data
    project_dir = Path(__file__).parent.parent.parent
    
    # Generate embeddings for training sequences
    train_label_file = project_dir / "Data" / "training_DBPs.txt"
    train_label_file_output = project_dir / "Data" / "training_DBPs_embedded.txt"
    
    if train_label_file.exists():
        print("\n" + "="*60)
        print("GENERATING TRAINING EMBEDDINGS")
        print("="*60)
        generate_embeddings_csv(
            str(train_label_file),
            str(train_label_file_output)
        )
    
    # Generate embeddings for test sequences
    test_label_file = project_dir / "Data" / "test_DBPs.txt"
    test_label_file_output = project_dir / "Data" / "test_DBPs_embedded.txt"
    
    if test_label_file.exists():
        print("\n" + "="*60)
        print("GENERATING TEST EMBEDDINGS")
        print("="*60)
        generate_embeddings_csv(
            str(test_label_file),
            str(test_label_file_output)
        )
