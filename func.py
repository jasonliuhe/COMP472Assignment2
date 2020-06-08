import pandas as pd


# Read data and separate data to training data and test set and four class
def read_data(dir_csv):
    csv_data = pd.read_csv(dir_csv)
    csv_data['Title'] = csv_data['Title'].str.lower()
    training_data = csv_data.loc[csv_data['year'] == 2018]
    test_set = csv_data.loc[csv_data['year'] == 2019]

    training_data_story = training_data.loc[training_data['Post Type'] == "story"]
    training_data_ask_hn = training_data.loc[training_data['Post Type'] == "ask_hn"]
    training_data_show_hn = training_data.loc[training_data['Post Type'] == "show_hn"]
    training_data_poll = training_data.loc[training_data['Post Type'] == "poll"]
    return training_data_story, training_data_ask_hn, training_data_show_hn, training_data_poll, test_set
