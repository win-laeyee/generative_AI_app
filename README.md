# Generative AI App

## Introduction

This README provides comprehensive information on a Generative AI application, powered by a fine-tuned Blenderbot model, which enables users to engage with the model through a Flask-based interface.

## Features

The Generative AI application allows users to:

1. **Ask Questions:** Users can input their questions or queries into the app to receive answers.
2. **Receive Answers:** The app uses the fine-tuned generative AI model to generate responses that are relevant to the user's questions.
3. **View Past Questions and Answers:** Users have the ability to review previous questions they've asked and the corresponding answers.

## Model Training and Evaluation

The core of the Generative AI application is the BlenderBot model, which has undergone fine-tuning. The training process is handled by the train_model_blenderbot.py script. The training process utilizes 'my_dataset.json,' a custom dataset created with personally curated information.

The evaluate.py script is designed to assess the performance of the generative AI model. It provides insights into how effectively the model can answer user questions and generate responses. The script identifies the best model and stores it in best_model folder.

## Deployment Steps
1. **Clone the repository:**

`git clone https://github.com/win-laeyee/generative_AI_app`

2. **Navigate to the Project Directory:**

`cd generative_AI_app`

3. **Set Up a Virtual Environment (Optional):**

Create a virtual environment to isolate project dependencies. 

* On macOS and Linux

`python3 -m venv venv`

`source venv/bin/activate`

* On Windows (Command Prompt):
`python -m venv venv`

`venv\Scripts\activate`

* On Windows (PowerShell):
`python -m venv venv`

`.\venv\Scripts\Activate.ps1`

4. **Install Flask and Dependencies:**

`pip install Flask`

`pip install -r requirements.txt`

5. **Run the Flask Application:**

`cd project`

`python3 train_model_blenderbot.py`

`python3 evaluate.py`

`python3 app.py`

## Project Structure

The project is structured as follows:

* `project/`: This folder contains the core application files.
  * `custom_dataset.py`: A custom data class for managing datasets.
  * `custom_collate.py`: Contains a custom_colalte function for data batching and padding.
  * `evaluate_model.py`: An evualate function using BLEU score
  * `train_model_blenderbot.py`: A script used for training the generative AI model, specifically the BlenderBot model.
  * `evaluate.py`: Script for evaluating the performance of the generative AI model.
  * `app.py`: The main application file responsible for handling user input and interactions.
  * `data/`: A directory for data management.
      * my_dataset.json: My dataset that was used to train and validate the generative AI model.
  * `fine_tuned_models/`: This directory contains fine-tuned models that are generated from `train_model_blenderbot.py`, and to be used in `evaluate.py`.
  * `best_model/`: This directory contains the best model that is generated from `evaluate.py`, and to be used in `app.py`.
  * `static/`: This directory contains static assets, particularly stylesheets for styling the app.
      * `css/`: Stylesheets.
        * `styles.css`: Used for styling the app's interface.
  * `templates/`: HTML templates for rendering user interfaces.
      * `question_form.html`: An HTML template for user input and question submission.
      * `response.html`: An HTML template for displaying AI responses.
* `README.md`
* `requirements.txt`: Contains a list of Python packages and their respective versions that are required for your project to run correctly.
