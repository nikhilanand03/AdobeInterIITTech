# Adobe Behaviour and Content Simulation Challenge

Communication in the digital realm is a dynamic interplay between senders and receivers, with each message crafted to evoke specific user behaviors. For marketers, the ultimate goal is to understand and predict user engagement – the likes, comments, shares, and purchases that define the success of their content.

In today's fast-paced digital landscape, businesses face the ever-growing challenge of delivering exceptional customer experiences. Adobe Experience Cloud, a comprehensive suite of tools, stands at the forefront, empowering businesses to design and deliver outstanding customer journeys. A seamless and engaging customer journey not only drives user satisfaction but also plays a pivotal role in brand differentiation, market credibility, and overall business success.

This challenge presents two critical aspects of customer engagement:

### Tweet Likes Prediction
Predicting tweet likes accurately is essential for understanding user engagement. Given the content of a tweet (text, company, username, media URLs, timestamp), predict its user engagement, specifically the number of likes.

### Tweet Content Generation
Predicting tweet likes accurately is essential for understanding user engagement. Given tweet metadata (company, username, media URL, timestamp), generate the tweet text. This task delves into the art and science of crafting compelling content that resonates with the audience.

The PS and datasets can be found here: https://drive.google.com/file/d/1ih79ljcUkxUaSa7nIvC73I5rjN5zMsw5/view

The work in this repository was done in a team of 6. I have specified my personal contributions in the comments beside each of the points.

## Repository Structure
1. Data preprocessing:
     - It contains 4 folders (EDA, Embeddings, Experiments, Preprocessing).
     - **EDA (Exploratory Data Analysis):** This section is divided into two Jupyter notebooks:
          - EDA_Stage1.ipynb: The first stage of exploratory data analysis. This was done during the initial stages of the hackathon.
          - EDA_Stage2.ipynb: The second stage of exploratory data analysis. This was done after correlations were deduced during the later stages of the hackathon, where we realised some additional features may be useful for the likes prediction. I worked on this.
     - **Embeddings:**
          - languagebind: Contains dependencies related to the LanguageBind embedder for embedding image, audio, and video. See [this](https://arxiv.org/abs/2310.01852) paper.
          - Audio_embeds.ipynb: Contains the code for generating audio embeddings from mp4 files attached with audio. I wrote this code.
          - EVA_CLIP.ipynb: A Jupyter notebook associated with EVA and CLIP embeddings.
          - Jina_Embeds.ipynb: Generating content text embeddings using the Jina.
          - Merge_embed.ipynb: Concatenating the embedding vectors.
          - extractLangBindEmbeddings.py: Contains functions to extract LanguageBind embeddings.
     - **Experiments:**
          - NeMO_ASR.ipynb: A Jupyter Notebook containing some experiments on transcribing speech audio. I worked on this, and played around with models of different sizes provided by NVIDIA's NeMO toolkit for speech-to-text synthesis. This allowed me to generate speech text in much lesser time.
          - keras_model.ipynb: A notebook containing experiments with Keras models.

3.
4.
5.


In the "train-test-split" folder, the bad video links are removed and the remaining data is split into 3 datasets ensuring no two reposts of the same post are in separate datasets (to avoid leakage).
