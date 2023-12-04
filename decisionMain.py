import music
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from decision_tree import DecisionTreeClassifier
from decision_tree import Node

def main(): 
    data= music.get_music()
    art_key=list(data[0]['artist'].keys())
    art_key[1]='artist_hotttnesss'   #rename the header since we have two hottness
    art_key[2]='artist_id'
    art_key[6]='artist_name'
    song_key=list(data[0]['song'].keys())
    song_key[8]='song_hotttnesss'
    song_key[9]='song_id'
    rel_key=list(data[0]['release'].keys())
    rel_key[0]='release_id'
    rel_key[1]='release_name'
    headers=art_key+rel_key+song_key
    all_row=[]
    for i in range(len(data)):
        art_val=list(data[i]['artist'].values())
        rel_val=list(data[i]['release'].values())
        song_val=list(data[i]['song'].values())
        each_row=art_val+rel_val+song_val
        all_row.append(each_row)
    
    filename = 'song.csv'
    with open(filename, 'w', newline="") as file:
        csvwriter = csv.writer(file) 
        csvwriter.writerow(headers)
        csvwriter.writerows(all_row)
    
    file_path = 'song.csv'
    df = pd.read_csv(file_path)
    print(df)

    #remove zero-number column

    columns_to_remove = ['location', 'similar', 'mode','title','latitude','longitude','artist_id','release_id','release_name','song_id','year']

    # Removing multiple columns
    df = df.drop(columns=columns_to_remove, axis=1)

    # Display the DataFrame after removing columns
    print(df)

    text_columns = []  # List to store columns with text data

    # Iterate through columns and check data types
    for column in df.columns:
        if df[column].dtype == object:  # Check if the data type is 'object' (usually represents text)
            text_columns.append(column)  # Add the column name to the list
            print(column)
    
    #mapping terms and artist_name to a unique machine readable value and throw the old string column
    label_encoder = LabelEncoder()
    df['terms_encode'] = label_encoder.fit_transform(df['terms'])
    df['artist_name_encode'] = label_encoder.fit_transform(df['artist_name'])
    columns_to_remove=['terms','artist_name']
    df = df.drop(columns=columns_to_remove, axis=1)
    print(df)

    first_row = df.head(1)
    first_row
    text_columns = []  # List to store columns with text data

    # Iterate through columns and check data types
    for column in df.columns:
        if df[column].dtype == object:  # Check if the data type is 'object' (usually represents text)
            text_columns.append(column)  # Add the column name to the list
            print(column)
    
    augmented_df = pd.concat([df] * 10, ignore_index=True)  # Duplicate DataFrame ten times
    noise = np.random.normal(0.05, 0.02, size=(len(augmented_df), len(augmented_df.columns)))  # Generate Gaussian noise 
    augmented_df = augmented_df + noise  # Add noise to the DataFrame

    augmented_df['familiarity_label'] = augmented_df['familiarity'].apply(lambda x: 1 if x > 0.5 else 0) # add labeled column and name it as familarity_label
    columns_to_remove=['familiarity']
    augmented_df = augmented_df.drop(columns=columns_to_remove, axis=1)
    print(augmented_df)

    X = augmented_df.drop(['familiarity_label'], axis=1)
    Y = augmented_df['familiarity_label']

    # Split the data into training and testing sets
    X = augmented_df.iloc[:500,:-1].values
    Y = augmented_df.iloc[:500,-1].values.reshape(-1,1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=6)
    classifier.fit(X_train,Y_train)
    classifier.print_tree()
    # print(X_train)

    # # Initialize the Decision Tree Classifier
    # clf = DecisionTreeClassifier(random_state=42)

    # # Fit the classifier on the training data
    # clf.fit(X_train, y_train)

    # # Make predictions on the test set
    # y_pred = clf.predict(X_test)

    # # Evaluate the classifier
    # accuracy = accuracy_score(y_test, y_pred)
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # class_report = classification_report(y_test, y_pred)

if __name__ == "__main__":
    main()