'''The main module which starts execution'''

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def read_into_dataframe(file_path: str) -> pd.DataFrame:
    '''Reads a csv into a dataframe'''
    return pd.read_csv(file_path)

def extract_data(df: pd.DataFrame, encoding: str="onehot") -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Returns a two pandas datadfs after feature extraction.
    The first one contains the independent variables and the second the dependent
    '''
    # removing duplicate data
    df.drop_duplicates(inplace=True)
    # converting age to a numeric value
    df["age"] = pd.to_numeric(df["age"].str.replace("'", ""), errors="coerce")
    # removing empty values
    # there are no empty values originally but converting age to a number created some
    df.dropna(inplace=True)

    # converting the different columns into a numerical value by means of one hot encoding
    category_cols = ["gender", "zipcodeOri", "merchant", "category"]
    if encoding == "ordinal":
        le = LabelEncoder()
        for column in category_cols:
            df[column] = le.fit_transform(df[column])

        return df[["age", "amount"] + category_cols], df["fraud"]
    elif encoding == "mean":
        for column in category_cols:
            mean = df.groupby(column)["fraud"].mean()
            df[column] = df[column].map(mean)

        return df[["age", "amount"] + category_cols], df["fraud"]

    new_columns = []
    for column in category_cols:
        one_hot = pd.get_dummies(df[column])
        new_columns.extend(one_hot.columns.to_list())
        # removing old column
        df.drop(column, axis=1, inplace=True)
        df = df.join(one_hot)

    return df[["step", "age","amount"] + new_columns], df["fraud"]

def train_model(x_train, y_train, solver="lbfgs") -> LogisticRegression:
    '''Trains and returns the model'''

    # using the LogisticRegressionModel
    print("Started training model")
    model = LogisticRegression(solver=solver)
    # model = svm.LinearSVC()
    model.fit(x_train, y_train)
    return model

if __name__ == "__main__":
    # first read the data and perform feature extraction/engineering
    df = read_into_dataframe("./data/bs140513_032310.csv")
    x_df, y_df = extract_data(df)

    # splitting data into testing an training
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2)

    model = train_model(x_train, y_train, "newton-cholesky")

    print(model.score(x_test, y_test))

    # diagnositcs = list(zip(model.coef_[0], model.feature_names_in_))
    # diagnositcs.sort()
    # print(diagnositcs)
