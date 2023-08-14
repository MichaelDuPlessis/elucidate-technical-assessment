'''The main module which starts execution'''

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def extract_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Returns a two pandas dataframes after feature extraction.
    The first one contains the independent variables and the second the dependent
    '''
    frame = pd.read_csv(file_path)
    # removing duplicate data
    frame.drop_duplicates(inplace=True)
    # converting age to a numeric value
    frame["age"] = pd.to_numeric(frame["age"].str.replace("'", ""), errors="coerce")
    # removing empty values
    # there are no empty values originally but converting age to a number created some
    frame.dropna(inplace=True)

    # converting the different columns into a numerical value by means of one hot encoding
    new_columns = []
    for column in ["gender", "zipcodeOri", "merchant", "category"]:
        one_hot = pd.get_dummies(frame[column])
        new_columns.extend(one_hot.columns.to_list())
        # removing old column
        frame.drop(column, axis=1, inplace=True)
        frame = frame.join(one_hot)


    return frame[["age","amount"] + new_columns], frame["fraud"]

if __name__ == "__main__":
    # first read the data and perform feature extraction/engineering
    x_df, y_df = extract_data("./data/bs140513_032310.csv")
    # splitting data into testing an training
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2)

    # using the LogisticRegressionModel
    model = LogisticRegression()
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
