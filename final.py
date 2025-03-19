import sys
import json
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB
from joblib import dump, load

special_char = [
    '.', ',', '"', '?', '!', ':', ';',
    '(', ')', '{', '}', '[', ']',
    '+', '-', '*', '/', '=', '%', '^', '<', '>', 
    '&', '|', '\\', '~', '#', '@', '_', 
    '£', '€', '¥',  "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"
    # Add more here as needed
]

def remove_special_characters(text):
    cleaned_text = ''.join(char for char in text if char not in special_char)
    
    return cleaned_text

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data
# Import necessary libraries for model training and inference

tokenizer = r'\S+'

def train_model(path_to_data_json, path_to_save):
    # Load data
    train_data = load_data(f"{path_to_data_json}/train.json")
    train_data_1 = load_data(f"{path_to_data_json}/valid_new.json")
    
    train_data = train_data + train_data_1
    

    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, use_idf=False, smooth_idf=False, norm='l1',   preprocessor=remove_special_characters, token_pattern=tokenizer)
    X_train = vectorizer.fit_transform([sample['text'] for sample in train_data])
    y_train = [sample['langid'] for sample in train_data]
    
    model = MultinomialNB(alpha = 0.00005)
    # model = ComplementNB(alpha=0.003, class_prior=None, fit_prior=True, norm=False)
    model.fit(X_train, y_train)
    
    model_path = f"{path_to_save}/model.joblib"
    vectorizer_path = f"{path_to_save}/vectorizer.joblib"

    dump(model, model_path)
    dump(vectorizer, vectorizer_path)
    print(f"Model and vectorizer saved to {path_to_save}")

    
    

def run_inference(path_to_save, path_to_test_json, output_path):
    # Load the saved model and vectorizer
    print(path_to_save)
    print(path_to_test_json)
    model = load(f"{path_to_save}/model.joblib")
    vectorizer = load(f"{path_to_save}/vectorizer.joblib")
    
    # Load test data
    def load_data(path):
        with open(path, 'r') as file:
            return json.load(file)
    
    test_data = load_data(path_to_test_json)
    
    # Transform the test data using the loaded vectorizer
    X_test = vectorizer.transform([sample['text'] for sample in test_data])
    
    # Predict the language IDs
    predictions = model.predict(X_test)
    
    # Write the predicted language IDs to the output file
    with open(output_path, 'w') as file:
        for pred in predictions:
            file.write(f"{pred}\n")
    
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    mode = sys.argv[1]
    
    if mode == 'train':
        _, _, path_to_data_json, path_to_save = sys.argv
        train_model(path_to_data_json, path_to_save)
    elif mode == 'test':
        _, _, path_to_save, path_to_test_json, output_path = sys.argv
        run_inference(path_to_save, path_to_test_json, output_path)
