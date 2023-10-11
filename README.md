# Generative AI App

## Introduction

This README provides comprehensive information on a Generative AI application that allows users to interact with a generative AI model. Users can ask questions and receive relevant answers, as well as access past questions and answers.

## Features

The Generative AI application allows users to:

1. Ask Questions: Users can input their questions or queries into the app to receive answers.
2. Receive Answers: The app uses the fine-tuned generative AI model to generate responses that are relevant to the user's questions.
3. View Past Questions and Answers: Users have the ability to review previous questions they've asked and the corresponding answers.

# Model Training and Evaluation

The core of the Generative AI application is the BlenderBot model, which has undergone fine-tuning. The training process is handled by the train_model_blenderbot.py script. The training process utilizes 'my_dataset.json,' a custom dataset created with personally curated information.

The evaluate.py script is designed to assess the performance of the generative AI model. It provides insights into how effectively the model can answer user questions and generate responses. The script identifies the best model and stores it in best_model folder.

# Deployment Steps
