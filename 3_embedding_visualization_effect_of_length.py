#%%
import os

import openai
from dotenv import load_dotenv

load_dotenv()
#%% md
# # List of sentences
#%%
sentences_short = [
    'The king wore black clothes',
    'राजा ने काले कपड़े पहने।',
    '王は黒い服を着ていた。',
]

sentences_medium = [
    'The king ate his food and went ahead to his royal chambers to put on his black clothes.',
    'राजा ने अपना भोजन किया और अपने काले कपड़े पहनने के लिए अपने शाही कक्षों की ओर बढ़ गए।',
    '王は食事を終え、黒い服を着るために王室の部屋へと進みました。'
]

sentences_long = [
    "The king ate his food and went ahead to his royal chambers to put on his black clothes. He wanted to head out to the city among his people.",
    "राजा ने अपना भोजन किया और अपने काले कपड़े पहनने के लिए अपने शाही कक्षों की ओर बढ़ गए। वह अपने लोगों के बीच शहर में जाना चाहते थे।",
    "王は食事を終え、黒い服を着るために王室の部屋へと進みました。彼は民衆に交じって街へ出たいと思っていました。"
]

sentences_paragraph = [
    "The king ate his food and went ahead to his royal chambers to put on his black clothes. He slipped out through a hidden passage, avoiding the watchful eyes of the guards. In the bustling city market, he was no longer a king but just another face in the crowd. Only then could he truly hear the whispers and worries of his people.",
    "राजा ने अपना भोजन किया और अपने काले कपड़े पहनने के लिए अपने शाही कक्षों की ओर बढ़ गए। वह पहरेदारों की चौकस नज़रों से बचते हुए, एक गुप्त मार्ग से बाहर निकल गया। शहर के हलचल भरे बाज़ार में, वह अब राजा नहीं बल्कि भीड़ में एक और चेहरा था। तभी वह वास्तव में अपने लोगों की कानाफूसी और चिंताओं को सुन सका।",
    "王は食事を終え、黒い服を着るために王室の部屋へと進みました。彼は衛兵の監視の目を避け、隠された通路からそっと抜け出しました。賑やかな街の市場では、彼はもはや王ではなく、群衆の中の一人にすぎませんでした。そうして初めて、彼は民のささやきや悩みを真に聞くことができたのです。"
]

sentences = [sentences_short, sentences_medium, sentences_long, sentences_paragraph]
sentences
#%% md
# ## Generate embeddings
#%%
# get embeddings for each sentence using openai embedding model
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
embedding_model = 'text-embedding-3-large'

output_sentences = []
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
if len(openai_embeddings) == len(output_sentences) and len(openai_embeddings[0]) != 0:
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

length = (['short'] * 3 + ['medium'] * 3 + ['long'] * 3 + ['paragraph'] * 3)

df = pd.DataFrame()
df['sentence'] = output_sentences
df['length'] = length
df['openai_embeddings'] = openai_embeddings
df['pca_x'] = pca[:, 0]
df['pca_y'] = pca[:, 1]
df['pca_z'] = pca[:, 2]
df.head()
#%%
import plotly.express as px

fig = px.scatter_3d(
    data_frame=df,
    x='pca_x',
    y='pca_y',
    z='pca_z',
    text=output_sentences,
    color='length',
    title='3D PCA of sentences using {}'.format(embedding_model),
)
#%%
# Improve layout
fig.update_traces(textposition='top center')
fig.write_image('embedding_3d_effect_of_length.png')
fig.show()
#%%
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity

metric = cosine_similarity
metric_name = "cosine_similarity"
#%% md
# ## Distance between english and hindi short statements
#%%
true_embedding_distances_en_hi_short = metric([openai_embeddings[0]], [openai_embeddings[1]])
# find true distance between english and japanese short statements
true_embedding_distances_en_jp_short = metric([openai_embeddings[0]], [openai_embeddings[2]])
true_embedding_distances_en_hi_short, true_embedding_distances_en_jp_short
#%% md
# ### Distance between medium sentences
#%%
# find true distance between english and hindi short statements
true_embedding_distances_en_hi_medium = metric([openai_embeddings[3]], [openai_embeddings[4]])
# find true distance between english and japanese short statements
true_embedding_distances_en_jp_medium = metric([openai_embeddings[3]], [openai_embeddings[5]])
true_embedding_distances_en_hi_medium, true_embedding_distances_en_jp_medium
#%% md
# ### Distance between long sentences
#%%
# find true distance between english and hindi long statements
true_embedding_distances_en_hi_long = metric([openai_embeddings[6]], [openai_embeddings[7]])
# find true distance between english and japanese short statements
true_embedding_distances_en_jp_long = metric([openai_embeddings[6]], [openai_embeddings[8]])
true_embedding_distances_en_hi_long, true_embedding_distances_en_jp_long
#%% md
# ### Distance between paragraphs
#%%
# find true distance between english and hindi paragraphs statements
true_embedding_distances_en_hi_paragraph = metric([openai_embeddings[9]], [openai_embeddings[10]])
# find true distance between english and japanese short statements
true_embedding_distances_en_jp_paragraph = metric([openai_embeddings[9]], [openai_embeddings[11]])
true_embedding_distances_en_hi_paragraph, true_embedding_distances_en_jp_paragraph
#%% md
# ## Create dataframe to analyze distances
#%%
distance_df = pd.DataFrame({'Type': ['short', 'medium', 'long', 'paragraph'],
                            f'{metric_name} en_hi': [true_embedding_distances_en_hi_short[0][0],
                                               true_embedding_distances_en_jp_medium[0][0],
                                               true_embedding_distances_en_hi_long[0][0],
                                               true_embedding_distances_en_hi_paragraph[0][0]],
                            f'{metric_name} en_jp': [true_embedding_distances_en_jp_short[0][0],
                                               true_embedding_distances_en_jp_medium[0][0],
                                               true_embedding_distances_en_jp_long[0][0],
                                               true_embedding_distances_en_jp_paragraph[0][0]],
                            })
distance_df.head()
#%%
distance_chart = px.line(distance_df, x='Type', y=f'{metric_name} en_hi', text='Type', markers=True)
distance_chart.update_traces(textposition="top center")
distance_chart.write_image('cosine_similarity_en_hi_effect_of_length.png')
distance_chart.show()
#%%
distance_chart = px.line(distance_df, x='Type', y=f'{metric_name} en_jp', text='Type', markers=True)
distance_chart.update_traces(textposition="top center")
distance_chart.write_image('cosine_similarity_en_jp_effect_of_length.png')
distance_chart.show()
#%%
