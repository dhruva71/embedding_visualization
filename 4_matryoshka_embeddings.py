#%%
import os

import openai
from dotenv import load_dotenv

load_dotenv()
#%% md
# # List of input sentences
#%%
sentences = [
    'What did the King wear?',
    'The king wore black clothes',
    'What is a red panda?',
    'The red panda is a small mammal.',
]
#%% md
# # Generate embeddings
#%%
# get embeddings for each sentence using openai embedding model
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
embedding_model = 'text-embedding-3-large'

openai_embeddings = []
for sentence in sentences:
    response = client.embeddings.create(
        model=embedding_model,
        input=sentence
    )
    embedding = response.data[0].embedding
    openai_embeddings.append(embedding)
    print(f'Received embedding for sentence: {sentence}')
#%%
if len(openai_embeddings) == len(sentences) and len(openai_embeddings[0]) != 0:
    print(f'Received embeddings for {len(openai_embeddings)} sentences, embedding size: {len(openai_embeddings[0])}')
#%% md
# # Get cosine similarity between every pair
#%%
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity

metric = cosine_similarity
metric_name = "cosine_similarity"
#%% md
# ## Function to calculate similarities
#%%
def calculate_similarities(openai_embeddings_list: list) -> dict:
    output_dict = {}
    for idx1, sentence1 in enumerate(sentences):
        for idx2, sentence2 in enumerate(sentences[idx1 + 1:]):
            actual_index = idx1 + 1 + idx2
            openai_embedding1 = [openai_embeddings_list[idx1]]
            openai_embedding2 = [openai_embeddings_list[actual_index]]

            output_dict[f'pair_{idx1}_{actual_index}'] = metric(openai_embedding1, openai_embedding2)[0][0]
    return output_dict
#%%
similarities_full = calculate_similarities(openai_embeddings_list=openai_embeddings)
similarities_full
#%% md
# # Matryoshka embeddings
#%% md
# ## Now we take only the first 64,128,256,512 to compare embeddings
#%%
openai_embeddings_64 = [x[:64] for x in openai_embeddings]
openai_embeddings_128 = [x[:128] for x in openai_embeddings]
openai_embeddings_256 = [x[:256] for x in openai_embeddings]
openai_embeddings_512 = [x[:512] for x in openai_embeddings]
#%% md
# ### Just to verify, let's look at the length of one
#%%
len(openai_embeddings_128[0])
#%% md
# ## Let us recalculate similarities using less dimensions
#%% md
# ### 64
#%%
similarities_64 = calculate_similarities(openai_embeddings_64)
similarities_64
#%% md
# ### 128
#%%
similarities_128 = calculate_similarities(openai_embeddings_128)
similarities_128
#%% md
# ## 256
#%%
similarities_256 = calculate_similarities(openai_embeddings_256)
similarities_256
#%% md
# ## 512
#%%
similarities_512 = calculate_similarities(openai_embeddings_512)
similarities_512
#%% md
# # Build a dataframe to compare
#%%
import pandas as pd

combined_data = {
    'full': similarities_full,
    '64': similarities_64,
    '128': similarities_128,
    '256': similarities_256,
    '512': similarities_512,
}
df = pd.DataFrame(combined_data)
df.index.name = 'sentence_pair'
df
#%% md
# # Plot on a chart
#%%
import plotly.express as px

fig = px.line(
    df,
    title='Cosine Similarity vs. Embedding Dimensions',
    markers=True,
)
fig.show()
#%% md
# # Find percentage errors from full measurements
#%% md
# ## full vs 64 dims
#%%
df['error_full_64'] = df['full']-df['64']
df
#%%
df['error_full_128'] = df['full']-df['128']
df['error_full_256'] = df['full']-df['256']
df['error_full_512'] = df['full']-df['512']
df
#%%
errors_df = df[[
    'error_full_64',
    'error_full_128',
'error_full_256',
'error_full_512',]].copy()
errors_df
#%% md
# ## Plot the absolute value of errors
#%%
fig = px.line(
    errors_df.abs(),
    title='Errors vs. Embedding Dimensions',
    markers=True,
)
fig.show()
#%%
