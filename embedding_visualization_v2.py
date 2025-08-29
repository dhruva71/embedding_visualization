#%%
import os

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
#%%
sentences_en=['The king wore black clothes', 'The queen wore black clothes', 'The joker wore black clothes']
sentences_hi=['राजा ने काले कपड़े पहने थे', 'रानी ने काले कपड़े पहने थे', 'जोकर ने काले कपड़े पहने थे']
sentences_jp=['王は黒い服を着ていた', '女王は黒い服を着ていた', 'ジョーカーは黒い服を着ていた']
sentences = [sentences_en, sentences_hi, sentences_jp]
sentences
#%%
# get embeddings for each sentence using openai embedding model
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
embedding_model = 'text-embedding-3-large'

output_sentences=[]
openai_embeddings = []
for sentence_group in sentences:
    for sentence in sentence_group:
        output_sentences.append(sentence)
        response = client.embeddings.create(
            model=embedding_model,
            input=sentence
        )
        embedding = response.data[0].embedding
        openai_embeddings.append(embedding)
        print(f'Received embedding for sentence: {sentence}')
#%%
print(openai_embeddings[0])
#%%
if len(openai_embeddings) == len(sentences) and len(openai_embeddings[0]) != 0:
    print(f'Received embeddings for {len(openai_embeddings)} sentences, embedding size: {len(openai_embeddings[0])}')
#%% md
# ## Run PCA
#%%
from sklearn.decomposition import PCA

pca = PCA(n_components=3).fit_transform(openai_embeddings)
#%%
pca
#%% md
# ## Visualize using plotly
#%% md
# ## Build a dataframe
#%%
import pandas as pd

languages = ['en'] * len(sentences_en) + \
            ['hi'] * len(sentences_hi) + \
            ['jp'] * len(sentences_jp)

df = pd.DataFrame()
df['sentence'] = output_sentences
df['language']=languages
df['openai_embeddings'] = openai_embeddings
df['pca_x'] = pca[:,0]
df['pca_y'] = pca[:,1]
df['pca_z'] = pca[:,2]
df.head()
#%%
import plotly.express as px

fig = px.scatter_3d(
    data_frame=df,
    x='pca_x',
    y='pca_y',
    z='pca_z',
    text=output_sentences,
    color='language',
    title='3D PCA of sentences using {}'.format(embedding_model),
)
#%%
# Improve layout
fig.update_traces(textposition='top center')
fig.write_image('embedding_3d.png')
fig.show()
#%% md
# ## Analysis of distances
#%% md
# ## True distance
#%%
from sklearn.metrics.pairwise import cosine_distances

# cosine distance = 1 - cosine_similarity
# find true distance between english and hindi first statements
true_embedding_distances_en_hi = cosine_distances([openai_embeddings[0]], [openai_embeddings[3]])
true_embedding_distances_en_hi
#%%
# find true distance between english and japanese first statements
true_embedding_distances_en_jp = cosine_distances([openai_embeddings[0]], [openai_embeddings[6]])
true_embedding_distances_en_jp
#%% md
# ## PCA distances
#%%
# find pca distance between english and hindi first statements
pc_embedding_distances_en_hi = cosine_distances([pca[0]], [pca[3]])
pc_embedding_distances_en_hi
#%%
# find pca distance between english and japanese first statements
pca_embedding_distances_en_jp = cosine_distances([pca[0]], [pca[6]])
pca_embedding_distances_en_jp
#%%
