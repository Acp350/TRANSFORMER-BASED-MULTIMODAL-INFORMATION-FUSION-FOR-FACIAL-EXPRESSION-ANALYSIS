import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.Audio.FeatureExtractionWav2Vec.FeatureTraining 
import get_classifier, extract_posteriors, clean_df


def process_AUs_avg(embs_path):
    """
    Process and average the Action Unit (AU) embeddings.
    """
    files = [f for f in os.listdir(embs_path) if os.path.isfile(os.path.join(embs_path, f))]
    avg_embs = []

    for file in files:
        df = pd.read_csv(os.path.join(embs_path, file), sep=";")
        avg_embs.append(df.mean().to_frame().T.assign(video_name=file.split('.')[0]))

    avg_embs_df = pd.concat(avg_embs, ignore_index=True)
    avg_embs_df['actor'] = avg_embs_df['video_name'].str.extract('(\d+)$').astype(int)
    avg_embs_df['emotion'] = avg_embs_df['video_name'].str.extract('-(\d+)-').astype(int) - 1

    return avg_embs_df.drop(columns=['speech'])


def train_and_evaluate(X, y, model_number, model_param, norm_type, seed):
    """
    Train and evaluate the model.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Normalize data
        if norm_type in [0, 1]:
            scaler = MinMaxScaler() if norm_type == 0 else StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Train classifier
        classifier = get_classifier(model_number, model_param, seed)
        classifier.fit(X_train, y_train)

        # Evaluate classifier
        predictions = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))

    return np.mean(accuracies)


def main(AUs_dir, model_number, model_param, norm_type, seed=2020):
    avg_embs_df = process_AUs_avg(AUs_dir)
    X, y = clean_df(avg_embs_df)

    avg_accuracy = train_and_evaluate(X, y, model_number, model_param, norm_type, seed)
    print(f"Average Accuracy: {avg_accuracy:.3f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-AUs', '--AUs_dir', type=str, required=True, help='Path with the embeddings to train/test the models')
    

    args = parser.parse_args()

    main(args.AUs_dir, args.model_number, args.param, args.type_of_norm)
