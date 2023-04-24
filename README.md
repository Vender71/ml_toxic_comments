# Russian Toxicity Classifier

This is an app for identifying toxic comments in Russian. The app uses a Bart-based classifier that was fine-tuned based on colloquial Rubert. The model was trained on a dataset of toxic comments collected from 2ch.hk and a dataset of poisonous comments collected from ok.ru.

## Model
[Bart](https://huggingface.co/docs/transformers/model_doc/bart)

## Features

Input: a comment in Russian
Output: the degree of toxicity and neutrality of the comment

## Installation

- Clone the repository: "git clone git@github.com:Vender71/ml_toxic_comments.git"

- Install the required packages: pip install -r requirements.txt"

## Testing

- Run "python -m pytest" in the console while in the project directory.

## Team:

This app was developed by Stanislav Borisenko, Vladislav Onufrienko, and Vera Tsymbalova. If you have any questions or issues with the app, please feel free to contact us.
