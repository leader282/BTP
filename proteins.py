#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
from transformers import T5Tokenizer, T5Model
import requests
import pickle


# In[2]:


df1 = pd.read_csv('new_chembl_inhibit_drug_target.csv')
df2 = pd.read_csv('cancer_ppi_combined.csv')


# In[3]:


protein_list1 = df1['target_accession_number']

protein_list2 = df2['node1_uniprot_id']

protein_list3 = df2['node2_uniprot_id']

protein_list = pd.concat([protein_list1, protein_list2, protein_list3], ignore_index=True)

protein_list = protein_list.drop_duplicates()

print(len(protein_list))


# In[4]:


# Load the Prot5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
print("Tokenizer loaded")


# In[5]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = T5Model.from_pretrained("Rostlab/prot_t5_xl_uniref50").half().to(device)
print("Model loaded")


# In[ ]:


# Function to fetch protein sequence from UniProt
def fetch_uniprot_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        sequence = ''.join(response.text.splitlines()[1:])  # Skip the FASTA header
        return sequence
    else:
        print(f"Error fetching sequence for {uniprot_id}")
        return None

# Function to generate embeddings for a list of protein sequences
def get_protein_embeddings(uniprot_ids):
    embeddings_dict = {}  # Dictionary to cache embeddings
    num = 0

    for uniprot_id in uniprot_ids:
        num += 1
        if num%20 == 0:
            print(f"Processing uniprot id : {uniprot_id} number {num}")

        try:
            sequence = fetch_uniprot_sequence(uniprot_id)  # Fetch the protein sequence
            if sequence:
                # Tokenize the input sequence
                inputs = tokenizer(sequence, return_tensors="pt", padding=True).to(device)

                # Add decoder_input_ids: initialize with the pad token id
                decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device)

                # Forward pass with encoder input and decoder input
                with torch.no_grad():
                    outputs = model(input_ids=inputs['input_ids'], decoder_input_ids=decoder_input_ids)

                # Extract embeddings (e.g., from encoder output or mean pooling)
                embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling for simplicity
                embeddings_dict[uniprot_id] = embedding.squeeze(0).cpu().numpy()  # Cache the embedding
            else:
                embeddings_dict[uniprot_id] = [0] * 1024  # Default zero embedding for missing sequences
        except:
            print(f"Error processing uniprot id : {uniprot_id} number {num}")

    return embeddings_dict


# Generate the embeddings for the unique UniProt IDs
embeddings_dict = get_protein_embeddings(protein_list)


# In[ ]:


while True:
    missing_ids = [m_id for m_id in protein_list if m_id not in embeddings_dict]
    if len(missing_ids) == 0:
        break
    new_dict = get_protein_embeddings(missing_ids)
    embeddings_dict.update(new_dict)


# In[ ]:


with open("embeddings_dict.pkl", "wb") as f:
    pickle.dump(embeddings_dict, f)

print("Embeddings saved to embeddings_dict.pkl")
print("Done")