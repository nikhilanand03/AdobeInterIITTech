# Adobe Behaviour and Content Simulation Challenge

This repository contains my own personal documentation of the Adobe Behaviour and Content Simulation Challenge that I participated in as a part of a team of 6. This was a part of the Inter-IIT Tech Meet 12.0 conducted at IIT Madras, where we won the Bronze medal.

The PS and datasets can be found here: https://drive.google.com/file/d/1ih79ljcUkxUaSa7nIvC73I5rjN5zMsw5/view

I have specified my personal contributions in the comments beside each of the points.

## Description of Problem Statement

Communication in the digital realm is a dynamic interplay between senders and receivers, with each message crafted to evoke specific user behaviors. For marketers, the ultimate goal is to understand and predict user engagement – the likes, comments, shares, and purchases that define the success of their content.

In today's fast-paced digital landscape, businesses face the ever-growing challenge of delivering exceptional customer experiences. Adobe Experience Cloud, a comprehensive suite of tools, stands at the forefront, empowering businesses to design and deliver outstanding customer journeys. A seamless and engaging customer journey not only drives user satisfaction but also plays a pivotal role in brand differentiation, market credibility, and overall business success.

This challenge presents two critical aspects of customer engagement:

### Tweet Likes Prediction
Predicting tweet likes accurately is essential for understanding user engagement. Given the content of a tweet (text, company, username, media URLs, timestamp), predict its user engagement, specifically the number of likes.

### Tweet Content Generation
Predicting tweet likes accurately is essential for understanding user engagement. Given tweet metadata (company, username, media URL, timestamp), generate the tweet text. This task delves into the art and science of crafting compelling content that resonates with the audience.

## Repository Structure
1. *Data preprocessing:*
     - It contains 4 folders (EDA, Embeddings, Experiments, Preprocessing).
     - **EDA (Exploratory Data Analysis):** This section is divided into two Jupyter notebooks:
          - EDA_Stage1.ipynb: The first stage of exploratory data analysis. This was done during the initial stages of the hackathon.
          - EDA_Stage2.ipynb: The second stage of exploratory data analysis. I wrote this code after correlations were deduced during the later stages of the hackathon, where we realised some additional features may be useful for the likes prediction.
     - **Embeddings:**
          - languagebind: Contains dependencies related to the LanguageBind embedder for embedding image, audio, and video. See [this](https://arxiv.org/abs/2310.01852) paper.
          - Audio_embeds.ipynb: Contains the code that I wrote for generating audio embeddings from mp4 files attached with audio.
          - EVA_CLIP.ipynb: A Jupyter notebook associated with EVA and CLIP embeddings.
          - Jina_Embeds.ipynb: Generating content text embeddings using the Jina.
          - Merge_embed.ipynb: Concatenating the embedding vectors.
          - extractLangBindEmbeddings.py: Contains functions to extract LanguageBind embeddings.
     - **Experiments:**
          - NeMO_ASR.ipynb: A Jupyter Notebook containing some experiments on transcribing speech audio. While working on this, I played around with models of different sizes provided by NVIDIA's NeMO toolkit for speech-to-text synthesis. This allowed me to generate speech text in much lesser time.
          - keras_model.ipynb: A notebook containing experiments with Keras models.
     - **Preprocessing:**
          - Image and Video Captioning: A notebook containing the code for generating image and video captions.
          - Image OCR: A notebook containing code for running an OCR on image inputs to generate subtitle text.
          - Video OCR: A notebook containing code for running an OCR on video inputs to generate subtitle text.
          - extractSpeechText.py: This python file contains a function that I wrote to convert speech to text. This was done using NeMO's ASR models as shown in the "Experiments" subfolder previously mentioned. The model was optimised for efficiency and speed.
      
2. *Task 1:*
     - **Experiments:** This folder contains several experiments that we tried as a team, and I will focus on a few that I worked on.
          - prompt_log_0-100_model.ipynb:
               - The idea originally was to divide the data into bins based on the number of likes, and generate models for these individual bins.
               - This model was made only for points with between 0 and 100 likes. The natural log of the likes was taken and then the model was fit based on that.
               - For the model, RoBERTaForSequenceClassification was used and prompts were generated based on which the model was finetuned.
          - prompt_log_100-3K5_model.ipynb: Similar model for points within a range of 100 to 3.5K likes.
     - The final model used in the submission was on a similar thread. At this point we generated additional features in the EDA Stage 2 and made a simpler model using them, involving an initial XGBoost classifier, the results of which determine which regressor model to use.
  
3. *Task 2:*
     - **Analogical Retrievers:** This folder contains several files that I worked on for Task 2 in order to implement the retrieval of analogical tweets to the current tweet.
          - Basically, the assumption is that tweets that are from a similar time period, similar company and have similar number of likes must be relatively similar in content. This is because those 3 are the primary inputs. Therefore, the top K most similar tweets to the one in question from the training dataset must be useful information for generating tweet content.
          - analogical_retriever_train.ipynb: Assuming the data has an unseen time period but seen brand, 3 columns are added to the train data with the 3 nearest neighbour tweet contents. The "nearness" is measured in terms of similarity in number of likes after filtering based on the inferred company name.
          - seen_brands.json and unseen_brands.json: These json files contain all seen and unseen brands. They can be referred to.
          - unseen_brands.ipynb: For an unseen brand in the data, since we can't filter by brand name in the training set, we use a dictionary to categorize all companies and use this to filter the data. When an unknown brand is provided as a data point, we initially filter companies within the same category as the target company, and then find nearest neighbors based on date and likes.
     - **LSTM Baseline Model:** This contains files pertaining a baseline LSTM model that I worked on for Task 2 in its initial stages. It is essentially an image captioning model, with the exception that image embeddings are fed directly through a trainable linear layer rather than using a trainable ResNet. Embeddings were generated using ViT. Results were reasonable for certain examples while the BLEU score remained low for other examples.
     - **Experiments:**
          - UnispeechEmbeddingModifiedArch.ipynb: I developed a modified architecture for the UnispeechEmbedding model, since we had initially planned on creating our own multimodal embedder model where each embedder also had trainable weights. This architecture was meant only for the audio part of the data and we planned on making further models for the other modalities. The reason for working on this is that the goal is always to minimise the number of parameters through a modified architecture, which works well for our specific use case but is also time and space efficient.
          - The remaining files contain attempts at finetuning and modifying various models such as "Flan-T5", "Falcon-7B", "Mistral", and "Phi-1.5".
     - main.ipynb: The final finetuned model code incorporating analogies for content generation
