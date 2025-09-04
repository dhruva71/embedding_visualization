#%%
import os

import openai
from dotenv import load_dotenv

load_dotenv()
#%%
sentences_short = [
    'The king wore black clothes',
    'राजा ने काले कपड़े पहने।',
    '王は黒い服を着ていた。',
]

sentences_medium = [
    'The king wore elegant black clothes during the feast',
    'राजा ने भोज के दौरान आकर्षक काले कपड़े पहने।',
    '王は宴会で優雅な黒い服を着ていた。',
]

sentences_long = [
    "As the ancient clock in the dusty hall chimed thirteen, signaling the arrival of midnight, a lone, curious mouse, no bigger than a thimble, cautiously emerged from a crack in the crumbling plaster wall, its whiskers twitching with anticipation as it scurried across the creaking floorboards towards a forgotten crumb of cheese nestled beneath a velvet armchair, completely unaware of the shadowy cat's eyes watching it intently from the dark corners of the vast, silent room.",

    "जैसे ही धूल भरे हॉल में पुरानी घड़ी ने तेरह बजाए, आधी रात के आगमन का संकेत देते हुए, एक अकेला, जिज्ञासु चूहा, जो एक अँगुलिमाल से बड़ा नहीं था, सावधानी से टूटी हुई प्लास्टर की दीवार में एक दरार से बाहर निकला, उसकी मूँछें प्रत्याशा से फड़क रही थीं जैसे ही वह मखमली कुर्सी के नीचे छिपे हुए पनीर के एक भूले हुए टुकड़े की ओर चरमराती फर्शबोर्ड पर सरपट दौड़ा, विशाल, शांत कमरे के अंधेरे कोनों से उसे उत्सुकता से देख रही एक छायादार बिल्ली की आँखों से पूरी तरह अनजान था।",

    "埃っぽいホールの古い時計が深夜の到来を告げる13回を鳴り響かせると、指貫ほどの大きさしかない孤独で好奇心旺盛な一匹のネズミが、崩れかけた漆喰の壁のひび割れから用心深く現れ、広大な静かな部屋の暗い隅からじっと見つめている影の猫の目には全く気づかずに、ひび割れた床板を走り抜け、ベルベットのアームチェアの下に忘れ去られたチーズのかけら目指して、そのひげをぴくぴくさせながら急いで向かっていった。"
]
sentences_paragraph = [
    "Rohan stared out the window of his home in Guwahati, watching the relentless monsoon rain hammer down on the tin roof, creating a deafening but familiar rhythm. The air was thick with the scent of wet earth and jasmine from the neighbour's garden. He clutched his worn-out football, his heart aching to join his friends in the field, which he knew was now a series of muddy puddles. Just as the afternoon began to fade, the downpour softened to a drizzle and then stopped altogether. A warm, golden light broke through the clouds, making the drenched green landscape glitter. Without a second's hesitation, Rohan grabbed his ball, flung open the door, and ran out onto the glistening, rain-washed street, his spirits soaring higher than the misty hills surrounding the city.",

    "रोहन गुवाहाटी में अपने घर की खिड़की से बाहर देख रहा था, जहाँ लगातार हो रही मानसूनी बारिश टिन की छत पर बरसकर एक बहरा कर देने वाली लेकिन जानी-पहचानी धुन बना रही थी। हवा में गीली मिट्टी और पड़ोसी के बगीचे से आती चमेली की महक घुली हुई थी। उसने अपनी घिसी-पिटी फुटबॉल को कसकर पकड़ रखा था, उसका दिल मैदान में अपने दोस्तों के पास जाने के लिए तड़प रहा था, जो वह जानता था कि अब कीचड़ भरे गड्ढों में बदल गया होगा। जैसे ही दोपहर ढलने लगी, मूसलाधार बारिश हल्की बूँदा-बाँदी में बदली और फिर पूरी तरह से रुक गई। बादलों के बीच से एक गर्म, सुनहरी रोशनी निकली, जिससे भीगा हुआ हरा परिदृश्य जगमगा उठा। एक पल भी सोचे बिना, रोहन ने अपनी गेंद उठाई, दरवाज़ा खोला, और चमकती हुई, बारिश से धुली सड़क पर दौड़ पड़ा, उसका उत्साह शहर को घेरे हुए धुंधली पहाड़ियों से भी ऊँचा उड़ रहा था।",

    "グワハティの自宅の窓から、ロハンは外をじっと見つめていた。絶え間なく降るモンスーンの雨がトタン屋根を叩きつけ、耳をつんざくようでありながらも聞き慣れたリズムを刻んでいた。空気は湿った土の匂いと、隣家の庭のジャスミンの香りで満たされていた。彼は使い古したサッカーボールを握りしめ、今や泥の水たまりだらけになっているであろうグラウンドで友達と合流したくて心が痛んだ。午後が暮れ始めたちょうどその時、土砂降りは霧雨へと変わり、やがて完全に止んだ。暖かく金色の光が雲間から差し込み、雨に濡れた緑の風景をキラキラと輝かせた。一瞬のためらいもなく、ロハンはボールを掴んでドアを勢いよく開け、雨に洗われてきらめく通りへと駆け出した。彼の心は、街を囲む霧がかった丘よりも高く舞い上がっていた。"
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
fig.write_image('embedding_3d.png')
fig.show()
#%% md
# ## Analysis of distances
#%% md
# ## True distance
#%% md
# ## Short
#%%
from sklearn.metrics.pairwise import cosine_distances

# cosine distance = 1 - cosine_similarity
# find true distance between english and hindi short statements
true_embedding_distances_en_hi_short = cosine_distances([openai_embeddings[0]], [openai_embeddings[1]])
# find true distance between english and japanese short statements
true_embedding_distances_en_jp_short = cosine_distances([openai_embeddings[0]], [openai_embeddings[2]])
true_embedding_distances_en_hi_short, true_embedding_distances_en_jp_short
#%% md
# ### Distance between medium sentences
#%%
# find true distance between english and hindi short statements
true_embedding_distances_en_hi_medium = cosine_distances([openai_embeddings[3]], [openai_embeddings[4]])
# find true distance between english and japanese short statements
true_embedding_distances_en_jp_medium = cosine_distances([openai_embeddings[3]], [openai_embeddings[5]])
true_embedding_distances_en_hi_medium, true_embedding_distances_en_jp_medium
#%% md
# ### Distance between long sentences
#%%
# find true distance between english and hindi long statements
true_embedding_distances_en_hi_long = cosine_distances([openai_embeddings[6]], [openai_embeddings[7]])
# find true distance between english and japanese short statements
true_embedding_distances_en_jp_long = cosine_distances([openai_embeddings[6]], [openai_embeddings[8]])
true_embedding_distances_en_hi_long, true_embedding_distances_en_jp_long
#%% md
# ### Distance between paragraphs
#%%
# find true distance between english and hindi paragraphs statements
true_embedding_distances_en_hi_paragraph = cosine_distances([openai_embeddings[9]], [openai_embeddings[10]])
# find true distance between english and japanese short statements
true_embedding_distances_en_jp_paragraph = cosine_distances([openai_embeddings[9]], [openai_embeddings[11]])
true_embedding_distances_en_hi_paragraph, true_embedding_distances_en_jp_paragraph
#%%
## Create dataframe to analyze distances
#%%
distance_df = pd.DataFrame({'Type': ['short', 'medium', 'long', 'paragraph'],
                            'Distance en_hi': [true_embedding_distances_en_hi_short[0][0],
                                               true_embedding_distances_en_jp_medium[0][0],
                                               true_embedding_distances_en_hi_long[0][0],
                                               true_embedding_distances_en_hi_paragraph[0][0]],
                            'Distance en_jp': [true_embedding_distances_en_jp_short[0][0],
                                               true_embedding_distances_en_jp_medium[0][0],
                                               true_embedding_distances_en_jp_long[0][0],
                                               true_embedding_distances_en_jp_paragraph[0][0]],
                            })
distance_df.head()
#%%
distance_chart = px.line(distance_df, x='Type', y='Distance en_hi')
distance_chart.show()
#%%
distance_chart = px.line(distance_df, x='Type', y='Distance en_jp')
distance_chart.show()
#%%
