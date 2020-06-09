import string
from decimal import Decimal

import pandas as pd
from collections import Counter


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


def get_count(data):
    title_list = data.Title.tolist()
    # remove all the punctuation for every word, so we will not remove right word which with the punctuation in next
    # step
    punctuation = "!\"#$%&'()*+,./:;<=>?@[\]^`{|}~"
    for x in range(len(title_list)):
        title_list[x] = title_list[x].translate(str.maketrans('', '', punctuation))
    # split all the string in to word
    vocabulary_list = []

    for x in range(len(title_list)):
        vocabulary_list = vocabulary_list + title_list[x].split(" ")
    # remove all the word contained number and special characters, because the number is to specific, it does help in
    # frequency
    remove_alpha_list = [item for item in vocabulary_list if
                         not any((char.isalpha() or char == '-' or char == '_') for char in item)]
    remove_list = [item for item in vocabulary_list if any(char.isdigit() for char in item)]
    remove_list = remove_list + remove_alpha_list
    new_alpha_items = [item for item in vocabulary_list if
                       not any(not (char.isalpha() or char == '-' or char == '_') for char in item)]
    new_items = [item for item in new_alpha_items if not any(char.isdigit() for char in item)]

    vocabulary_list = new_items

    # remove empty string
    while "" in vocabulary_list:
        vocabulary_list.remove("")
    remove_list.append("")

    total_word = len(vocabulary_list)
    vocabulary_count = Counter(vocabulary_list)
    vocabulary_list_deduplication = list(dict.fromkeys(vocabulary_list))
    sorted_vocabulary_list = sorted(vocabulary_list_deduplication)

    return vocabulary_count, sorted_vocabulary_list, remove_list, total_word


def get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list, show_hn_vocabulary_list, poll_vocabulary_list):
    vocabulary_list = story_vocabulary_list + ask_hn_vocabulary_list + show_hn_vocabulary_list + poll_vocabulary_list
    vocabulary_list_deduplication = list(dict.fromkeys(vocabulary_list))
    return vocabulary_list_deduplication


def get_remove_list(story_remove_list, ask_hn_remove_list, show_hn_remove_list, poll_remove_list):
    remove_list = story_remove_list + ask_hn_remove_list + show_hn_remove_list + poll_remove_list
    remove_list_deduplication = list(dict.fromkeys(remove_list))
    return remove_list_deduplication


def save_file(file_name, list_name):
    with open(file_name, 'w') as f:
        for item in list_name:
            f.write("%s\n" % item)


def create_model(vocabulary_list, story_count, story_total_word, story_vocabulary_list, ask_hn_count,
                 ask_hn_total_word, ask_hn_vocabulary_list, show_hn_count, show_hn_total_word,
                 show_hn_vocabulary_list, poll_count, poll_total_word, poll_vocabulary_list, file_name):
    p_story, p_ask_hn, p_show_hn, p_poll = [], [], [], []
    for x in vocabulary_list:
        if len(story_vocabulary_list) > 0:
            p_story.append((Decimal(story_count[x]) + Decimal(0.5)) / (
                        Decimal(story_total_word) + Decimal(0.5) * len(story_vocabulary_list)))
        else:
            p_story.append(Decimal(0))

        if len(ask_hn_vocabulary_list) > 0:
            p_ask_hn.append((Decimal(ask_hn_count[x]) + Decimal(0.5)) / (
                        Decimal(ask_hn_total_word) + Decimal(0.5) * len(ask_hn_vocabulary_list)))
        else:
            p_ask_hn.append(Decimal(0))

        if len(show_hn_vocabulary_list) > 0:
            p_show_hn.append((Decimal(show_hn_count[x]) + Decimal(0.5)) / (
                        Decimal(show_hn_total_word) + Decimal(0.5) * len(show_hn_vocabulary_list)))
        else:
            p_show_hn.append(Decimal(0))

        if len(poll_vocabulary_list) > 0:
            p_poll.append((Decimal(poll_count[x]) + Decimal(0.5)) / (
                        Decimal(poll_total_word) + Decimal(0.5) * len(poll_vocabulary_list)))
        else:
            p_poll.append(Decimal(0))

    with open(file_name, 'w') as f:
        for x in range(len(vocabulary_list)):
            f.write("%s  %d  %5f  %d  %5f  %d  %5f  %d  %5f\n" %
                    (vocabulary_list[x], story_count[vocabulary_list[x]], p_story[x],
                     ask_hn_count[vocabulary_list[x]], p_ask_hn[x], show_hn_count[vocabulary_list[x]],
                     p_show_hn[x], poll_count[vocabulary_list[x]], p_poll[x]))

    return p_story, p_ask_hn, p_show_hn, p_poll

def test_Title(Title, p_story, p_ask_hn, p_show_hn, p_poll, store_p, ask_hn_p, show_hn_p, poll_p, o_vocabulary_list):
    # remove all the punctuation for every word, so we will not remove right word which with the punctuation in next
    # step
    punctuation = "!\"#$%&'()*+,./:;<=>?@[\]^`{|}~"
    Title = Title.translate(str.maketrans('', '', punctuation))
    # split all the string in to word
    vocabulary_list = Title.split(" ")
    remove_alpha_list = [item for item in vocabulary_list if
                         not any((char.isalpha() or char == '-' or char == '_') for char in item)]
    remove_list = [item for item in vocabulary_list if any(char.isdigit() for char in item)]
    remove_list = remove_list + remove_alpha_list
    new_alpha_items = [item for item in vocabulary_list if
                       not any(not (char.isalpha() or char == '-' or char == '_') for char in item)]
    new_items = [item for item in new_alpha_items if not any(char.isdigit() for char in item)]

    vocabulary_list = new_items
    store_score = Decimal(store_p).log10()
    ask_hn_score = Decimal(ask_hn_p).log10()
    show_hn_score = Decimal(show_hn_p).log10()
    poll_score = Decimal(poll_p).log10()

    for vocabulary in vocabulary_list:
        if vocabulary in o_vocabulary_list:
            index = o_vocabulary_list.index(vocabulary)
            store_score += p_story[index]
            ask_hn_score += p_ask_hn[index]
            show_hn_score += p_show_hn[index]
            poll_score += p_poll[index]

    return store_score, ask_hn_score, show_hn_score, poll_score


def test_model(vocabulary_list, test_set, p_story, p_ask_hn, p_show_hn, p_poll, store_p, ask_hn_p, show_hn_p, poll_p):
    Title = test_set.Title.tolist()
    Type = test_set["Post Type"]
    type_list = Type.tolist()
    f = open("baselineresult.txt", 'w')
    count = 0
    wrong_num = 0
    for x in Title:
        story_score, ask_hn_score, show_hn_score, poll_score = test_Title(x, p_story, p_ask_hn, p_show_hn, p_poll, store_p, ask_hn_p, show_hn_p, poll_p, vocabulary_list)
        score = {"story": story_score, "ask_hn": ask_hn_score, "show_hn": show_hn_score, "poll": poll_score}
        check = "right"
        if max(score, key=score.get) != type_list[count]:
            check = "wrong"
            wrong_num += 1
        f.write("%d  %s  %s  %5f  %5f  %5f  %5f  %s  %s\n" % (count, x, type_list[count], story_score, ask_hn_score, show_hn_score, poll_score, max(score, key=score.get), check))
        count += 1
    f.close()
    print(wrong_num)

