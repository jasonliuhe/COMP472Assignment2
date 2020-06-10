from decimal import Decimal
import func


def main():
    training_data_story, training_data_ask_hn, training_data_show_hn, training_data_poll, test_set = \
        func.read_data('Data/hns_2018_2019.csv')
    story_count, story_vocabulary_list, story_remove_list, story_total_word = \
        func.get_count(training_data_story)
    ask_hn_count, ask_hn_vocabulary_list, ask_hn_remove_list, ask_hn_total_word = \
        func.get_count(training_data_ask_hn)
    show_hn_count, show_hn_vocabulary_list, show_hn_remove_list, show_hn_total_word = \
        func.get_count(training_data_show_hn)
    poll_count, poll_vocabulary_list, poll_remove_list, poll_total_word = \
        func.get_count(training_data_poll)

    total_word = story_total_word + ask_hn_total_word + show_hn_total_word + poll_total_word
    store_p, ask_hn_p, show_hn_p, poll_p = Decimal(story_total_word)/Decimal(total_word), \
                                           Decimal(ask_hn_total_word)/Decimal(total_word), \
                                           Decimal(show_hn_total_word)/Decimal(total_word), \
                                           Decimal(poll_total_word)/Decimal(total_word)
    vocabulary_list = func.get_vocabulary_list(story_vocabulary_list, ask_hn_vocabulary_list,
                                               show_hn_vocabulary_list, poll_vocabulary_list)

    remove_list = func.get_remove_list(story_remove_list, ask_hn_remove_list,
                                       show_hn_remove_list, poll_remove_list)

    func.save_file('vocabulary.txt', vocabulary_list)
    func.save_file('remove.txt', remove_list)

    p_story, p_ask_hn, p_show_hn, p_poll = func.create_model(vocabulary_list, story_count, story_total_word,
                                                             story_vocabulary_list, ask_hn_count, ask_hn_total_word,
                                                             ask_hn_vocabulary_list, show_hn_count, show_hn_total_word,
                                                             show_hn_vocabulary_list, poll_count, poll_total_word,
                                                             poll_vocabulary_list, 'model-2018.txt')
    func.test_model(vocabulary_list, test_set, p_story, p_ask_hn, p_show_hn, p_poll, store_p, ask_hn_p, show_hn_p, poll_p)


def Experiment1():
    training_data_story, training_data_ask_hn, training_data_show_hn, training_data_poll, test_set = \
        func.read_data('Data/hns_2018_2019.csv')
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

    func.save_file('E1vocabulary.txt', vocabulary_list)
    func.save_file('E1remove.txt', remove_list)

    p_story, p_ask_hn, p_show_hn, p_poll = func.create_model(vocabulary_list, story_count, story_total_word,
                                                             story_vocabulary_list, ask_hn_count, ask_hn_total_word,
                                                             ask_hn_vocabulary_list, show_hn_count, show_hn_total_word,
                                                             show_hn_vocabulary_list, poll_count, poll_total_word,
                                                             poll_vocabulary_list, 'E1model-2018.txt')
    func.test_model(vocabulary_list, test_set, p_story, p_ask_hn, p_show_hn, p_poll, store_p, ask_hn_p, show_hn_p,
                    poll_p)


# main()
Experiment1()
