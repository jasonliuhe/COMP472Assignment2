from decimal import Decimal
import func
import matplotlib.pyplot as plt

data = 'Data/hns_2018_2019.csv'


def main():
    training_data_story, training_data_ask_hn, training_data_show_hn, training_data_poll, test_set = \
        func.read_data(data)
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.get_count(training_data_story)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.get_count(training_data_ask_hn)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.get_count(training_data_show_hn)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.get_count(training_data_poll)

    total_word = story_total_word + ask_hn_total_word + show_hn_total_word + poll_total_word
    store_p, ask_hn_p, show_hn_p, poll_p = Decimal(story_total_word) / Decimal(total_word), \
                                           Decimal(ask_hn_total_word) / Decimal(total_word), \
                                           Decimal(show_hn_total_word) / Decimal(total_word), \
                                           Decimal(poll_total_word) / Decimal(total_word)
    vocabulary_list = func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                               show_hn_vocabulary_list, poll_vocabulary_list)

    remove_list = func.get_remove_list(story_remove_list, ask_hn_remove_list,
                                       show_hn_remove_list, poll_remove_list)

    func.save_file('baseline/vocabulary.txt', vocabulary_list)
    func.save_file('baseline/remove.txt', remove_list)

    p_story, p_ask_hn, p_show_hn, p_poll = func.create_model(vocabulary_list, story_count, story_total_word,
                                                             story_vocabulary_list, ask_hn_count, ask_hn_total_word,
                                                             ask_hn_vocabulary_list, show_hn_count, show_hn_total_word,
                                                             show_hn_vocabulary_list, poll_count, poll_total_word,
                                                             poll_vocabulary_list, 'baseline/model-2018.txt')
    func.test_model(vocabulary_list, test_set, p_story, p_ask_hn, p_show_hn, p_poll, store_p, ask_hn_p, show_hn_p,
                    poll_p, "baseline/baselineresult.txt")


def Experiment1():
    training_data_story, training_data_ask_hn, training_data_show_hn, training_data_poll, test_set = \
        func.read_data(data)
    stop_words_list = func.read_stop_words('Data/stopwords.txt')
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.e1_get_count(training_data_story, stop_words_list)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.e1_get_count(training_data_ask_hn, stop_words_list)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.e1_get_count(training_data_show_hn, stop_words_list)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.e1_get_count(training_data_poll, stop_words_list)

    total_word = story_total_word + ask_hn_total_word + show_hn_total_word + poll_total_word
    store_p, ask_hn_p, show_hn_p, poll_p = Decimal(story_total_word) / Decimal(total_word), \
                                           Decimal(ask_hn_total_word) / Decimal(total_word), \
                                           Decimal(show_hn_total_word) / Decimal(total_word), \
                                           Decimal(poll_total_word) / Decimal(total_word)
    vocabulary_list = func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                               show_hn_vocabulary_list, poll_vocabulary_list)

    remove_list = func.get_remove_list(story_remove_list, ask_hn_remove_list,
                                       show_hn_remove_list, poll_remove_list)

    func.save_file('stopword/stopword_vocabulary.txt', vocabulary_list)
    func.save_file('stopword/stopword_remove.txt', remove_list)

    p_story, p_ask_hn, p_show_hn, p_poll = func.create_model(vocabulary_list, story_count, story_total_word,
                                                             story_vocabulary_list, ask_hn_count, ask_hn_total_word,
                                                             ask_hn_vocabulary_list, show_hn_count, show_hn_total_word,
                                                             show_hn_vocabulary_list, poll_count, poll_total_word,
                                                             poll_vocabulary_list, 'stopword/stopword-model.txt')
    func.test_model(vocabulary_list, test_set, p_story, p_ask_hn, p_show_hn, p_poll, store_p, ask_hn_p, show_hn_p,
                    poll_p, "stopword/stopword-result.txt")


def Experiment2():
    training_data_story, training_data_ask_hn, training_data_show_hn, training_data_poll, test_set = \
        func.read_data('Data/hns_2018_2019.csv')
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.e2_get_count(training_data_story)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.e2_get_count(training_data_ask_hn)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.e2_get_count(training_data_show_hn)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.e2_get_count(training_data_poll)

    total_word = story_total_word + ask_hn_total_word + show_hn_total_word + poll_total_word
    store_p, ask_hn_p, show_hn_p, poll_p = Decimal(story_total_word) / Decimal(total_word), \
                                           Decimal(ask_hn_total_word) / Decimal(total_word), \
                                           Decimal(show_hn_total_word) / Decimal(total_word), \
                                           Decimal(poll_total_word) / Decimal(total_word)
    vocabulary_list = func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                               show_hn_vocabulary_list, poll_vocabulary_list)

    remove_list = func.get_remove_list(story_remove_list, ask_hn_remove_list,
                                       show_hn_remove_list, poll_remove_list)

    func.save_file('wordlength/wordlength_vocabulary.txt', vocabulary_list)
    func.save_file('wordlength/wordlength_remove.txt', remove_list)

    p_story, p_ask_hn, p_show_hn, p_poll = func.create_model(vocabulary_list, story_count, story_total_word,
                                                             story_vocabulary_list, ask_hn_count, ask_hn_total_word,
                                                             ask_hn_vocabulary_list, show_hn_count, show_hn_total_word,
                                                             show_hn_vocabulary_list, poll_count, poll_total_word,
                                                             poll_vocabulary_list, 'wordlength/wordlength-model.txt')
    func.test_model(vocabulary_list, test_set, p_story, p_ask_hn, p_show_hn, p_poll, store_p, ask_hn_p, show_hn_p,
                    poll_p, "wordlength/wordlength-result.txt")


def Experiment3_1():
    # part 1
    threshold1 = 1
    training_data_story, training_data_ask_hn, training_data_show_hn, training_data_poll, test_set = \
        func.read_data(data)
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.e3_1_get_count(training_data_story, threshold1)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.e3_1_get_count(training_data_ask_hn, threshold1)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.e3_1_get_count(training_data_show_hn, threshold1)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.e3_1_get_count(training_data_poll, threshold1)
    len_vocabulary_list_1 = len(func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                                         show_hn_vocabulary_list, poll_vocabulary_list))

    threshold5 = 5
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.e3_1_get_count(training_data_story, threshold5)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.e3_1_get_count(training_data_ask_hn, threshold5)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.e3_1_get_count(training_data_show_hn, threshold5)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.e3_1_get_count(training_data_poll, threshold5)
    len_vocabulary_list_5 = len(func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                                         show_hn_vocabulary_list, poll_vocabulary_list))

    threshold10 = 10
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.e3_1_get_count(training_data_story, threshold10)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.e3_1_get_count(training_data_ask_hn, threshold10)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.e3_1_get_count(training_data_show_hn, threshold10)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.e3_1_get_count(training_data_poll, threshold10)
    len_vocabulary_list_10 = len(func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                                          show_hn_vocabulary_list, poll_vocabulary_list))

    threshold15 = 15
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.e3_1_get_count(training_data_story, threshold15)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.e3_1_get_count(training_data_ask_hn, threshold15)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.e3_1_get_count(training_data_show_hn, threshold15)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.e3_1_get_count(training_data_poll, threshold15)
    len_vocabulary_list_15 = len(func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                                          show_hn_vocabulary_list, poll_vocabulary_list))

    # plot for
    plt.plot([threshold1, threshold5, threshold10, threshold15],
             [len_vocabulary_list_1, len_vocabulary_list_5, len_vocabulary_list_10, len_vocabulary_list_15], 'ro')
    plt.axis([0, 20, 0,
              max(len_vocabulary_list_1, len_vocabulary_list_5, len_vocabulary_list_10, len_vocabulary_list_15) + 50])
    plt.ylabel('number of vocabulary left')
    plt.xlabel('threshold')
    plt.title("1")
    plt.show()


def Experiment3_2():
    # part 2
    threshold5 = 0.05
    training_data_story, training_data_ask_hn, training_data_show_hn, training_data_poll, test_set = \
        func.read_data(data)
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.e3_2_get_count(training_data_story, threshold5)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.e3_2_get_count(training_data_ask_hn, threshold5)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.e3_2_get_count(training_data_show_hn, threshold5)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.e3_2_get_count(training_data_poll, threshold5)
    len_vocabulary_list_5 = len(func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                                         show_hn_vocabulary_list, poll_vocabulary_list))

    threshold10 = 0.1
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.e3_2_get_count(training_data_story, threshold10)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.e3_2_get_count(training_data_ask_hn, threshold10)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.e3_2_get_count(training_data_show_hn, threshold10)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.e3_2_get_count(training_data_poll, threshold10)
    len_vocabulary_list_10 = len(func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                                          show_hn_vocabulary_list, poll_vocabulary_list))

    threshold15 = 0.15
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.e3_2_get_count(training_data_story, threshold15)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.e3_2_get_count(training_data_ask_hn, threshold15)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.e3_2_get_count(training_data_show_hn, threshold15)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.e3_2_get_count(training_data_poll, threshold15)
    len_vocabulary_list_15 = len(func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                                          show_hn_vocabulary_list, poll_vocabulary_list))

    threshold20 = 0.2
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.e3_2_get_count(training_data_story, threshold20)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.e3_2_get_count(training_data_ask_hn, threshold20)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.e3_2_get_count(training_data_show_hn, threshold20)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.e3_2_get_count(training_data_poll, threshold20)
    len_vocabulary_list_20 = len(func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                                          show_hn_vocabulary_list, poll_vocabulary_list))

    threshold25 = 0.25
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.e3_2_get_count(training_data_story, threshold25)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.e3_2_get_count(training_data_ask_hn, threshold25)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.e3_2_get_count(training_data_show_hn, threshold25)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.e3_2_get_count(training_data_poll, threshold25)
    len_vocabulary_list_25 = len(func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                                          show_hn_vocabulary_list, poll_vocabulary_list))

    plt.plot([threshold5, threshold10, threshold15, threshold20, threshold25],
             [len_vocabulary_list_5, len_vocabulary_list_10, len_vocabulary_list_15, len_vocabulary_list_20,
              len_vocabulary_list_25], 'ro')
    plt.axis([0, 1, 0, max(len_vocabulary_list_5, len_vocabulary_list_10, len_vocabulary_list_15) + 50])
    plt.ylabel('number of vocabulary left')
    plt.xlabel('threshold')
    plt.title("2")
    plt.show()


main()
Experiment1()
Experiment2()
Experiment3_1()
Experiment3_2()
