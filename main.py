import pandas as pd
import func


def main():
    training_data_story, training_data_ask_hn, training_data_show_hn, training_data_poll, test_set = func.read_data('Data/hns_2018_2019.csv')
    print(training_data_story.Title[0])


main()
