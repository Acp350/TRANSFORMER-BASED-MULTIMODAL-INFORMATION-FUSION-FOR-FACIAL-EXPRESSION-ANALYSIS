import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import classifiers  
import data_processing as dp  

def main(args):
    # Set random seed for reproducibility
    np.random.seed(2020)

    # Load and preprocess data
    data_path = os.path.join(args.embs_dir.rsplit("/", 1)[0], "avg_embs_512.csv")
    df = dp.load_data(data_path, args.embs_dir)

    # Normalize data if required
    if args.type_of_norm != 2:
        df = dp.normalize_data(df, norm_type=args.type_of_norm)

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('emotion', axis=1), df['emotion'], test_size=0.2, random_state=2020)

    # Initialize and train the classifier
    model = classifiers.get_classifier(args.model_number, args.param)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Save model outputs if specified
    if args.out_dir:
        dp.save_outputs(model, X_train, y_train, X_test, y_test, args.out_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    

    args = parser.parse_args()
    main(args)
