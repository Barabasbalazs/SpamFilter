import os
from string import punctuation
from math import log

def get_stopwords():
    file = open("stopwords.txt", "r")
    stop_words_arr = []
    for line in file:
        stop_words_arr.append(line.split()[0])
    file.close()

    file = open("stopwords2.txt", "r")
    for line in file:
        stop_words_arr.append(line.split()[0])
    file.close()

    for i in punctuation:
        stop_words_arr.append(i)

    return stop_words_arr


def tokanize_document(stop_words, email):
    cleaned_email = []
    for word in email:
        if word in stop_words:
            continue
        cleaned_email.append(word)
    return cleaned_email


def make_histogram(doc):
    hist = {}
    for word in doc:
        if word in hist:
            hist[word] += 1
        else:
            hist[word] = 1
    return hist


def read_in_emails(filename, stop_words):

    file1 = open(filename, "r", encoding = "utf-8")

    emails = []
    file_names = []
    for name in file1:
        conc_name = name.split()[0]
        file_names.append(conc_name)

    file1.close()

    os.chdir("spam")

    for name_cat in file_names:
        if name_cat in os.listdir():
            file = open(name_cat, "r", encoding = "utf-8", errors = "ignore")
            words_array = []
            for line in file:
                if "Subject:" in line.split():
                    continue
                words_in_line = line.split()
                for word in words_in_line:
                    words_array.append(word.lower())
            tokenized_words = tokanize_document(stop_words, words_array)
            emails.append((name_cat, make_histogram(tokenized_words)))
            file.close()
    
    os.chdir("../ham")

    for name_cat in file_names:
        if name_cat in os.listdir():
            file = open(name_cat, "r", encoding = "utf-8", errors = "ignore")
            words_array = []
            for line in file:
                if "Subject:" in line.split():
                    continue
                words_in_line = line.split()
                for word in words_in_line:
                    words_array.append(word.lower())
            tokenized_words = tokanize_document(stop_words, words_array)
            emails.append((name_cat, make_histogram(tokenized_words)))
            file.close()

    os.chdir("..")

    return emails
            

def count_cat_of_mail(mails):
    spams = 0
    hams = 0
    for name in mails:
        if "ham" in name[0]:
            hams += 1
        else:
            spams += 1
    return (hams / len(mails), spams / len(mails))


def dict_of_cat(mails):
    ham_dict = {}
    spam_dict = {}
    for mail in mails:
        if "ham" in mail[0]:
            for word in mail[1]:
                if word in ham_dict:
                    ham_dict[word] += 1
                else:
                    ham_dict[word] = 1
        else:
            for word in mail[1]:
                if word in spam_dict:
                    spam_dict[word] += 1
                else:
                    spam_dict[word] = 1
    return ham_dict, spam_dict


def count_of_words_in_cat(ham_dict, spam_dict):
    h_count = 0
    s_count = 0

    for key in ham_dict:
        h_count += ham_dict[key]
    for key in spam_dict:
        s_count += spam_dict[key]

    return h_count, s_count

def label_document(document, ham_p, spam_p, ham_dict, spam_dict, h_count, s_count, lamda, alpha, v_num):
    p_sum = 0
    for word in document:
        if word in spam_dict:
            # wk_spam = spam_dict[word] / s_count
            wk_spam = (spam_dict[word] + alpha) / ((alpha * v_num) + s_count)
        else:
            wk_spam = lamda

        if word in ham_dict:
            # wk_ham = ham_dict[word] / h_count
            wk_ham = (ham_dict[word] + alpha) / ((alpha * v_num) + h_count)
        else:
            wk_ham = lamda

        if wk_spam < lamda:
            wk_spam = lamda
        if wk_ham < lamda:    
            wk_ham = lamda
    
        p_sum += document[word] * (log(wk_spam) - log(wk_ham))

    l = (log(spam_p) - log(ham_p)) + p_sum
    return l
    # r = (spam_p / ham_p) * (p_mul_spam / p_mul_ham)


def universal_word_count(dict1, dict2):
    count = len(dict1)
    for word in dict2:
        if word not in dict1:
            count += 1
    return count


def count_spam(mails):
    count = 0
    for word in mails:
        if "spam" in word[0]:
            count += 1
    return count


def model_results(training_mails, test_mails, ham_p, spam_p, ham_dict, spam_dict, h_count, s_count):
    lamda = 0.000001
    alpha = 0.01

    tr_count = 0
    te_count = 0
    tr_false_assumptions = [0, 0]
    te_false_assumptions = [0, 0]

    v_num = universal_word_count(ham_dict, spam_dict)

    nr_of_spam = []
    nr_of_spam.append(count_spam(training_mails))
    nr_of_spam.append(count_spam(test_mails))

    for mail in training_mails:
        res = label_document(mail[1], ham_p, spam_p, ham_dict, spam_dict, h_count, s_count, lamda, alpha, v_num)
        if res > 0 and "spam" in mail[0]:
            tr_count += 1
        elif res <= 0 and "ham" in mail[0]:
            tr_count += 1
        elif res > 0 and "ham" in mail[0]:
            tr_false_assumptions[0] += 1
        elif res <= 0 and "spam" in mail[0]:
            tr_false_assumptions[1] += 1

    for mail in test_mails:
        res = label_document(mail[1], ham_p, spam_p, ham_dict, spam_dict, h_count, s_count, lamda, alpha, v_num)
        if res > 0 and "spam" in mail[0]:
            te_count += 1
        elif res <= 0 and "ham" in mail[0]:
            te_count += 1
        elif res > 0 and "ham" in mail[0]:
            te_false_assumptions[0] += 1
        elif res <=0 and "spam" in mail[0]:
            te_false_assumptions[1] += 1

    file = open("result.txt", "w")

    file.write("Lamda is:{} \n".format(lamda))
    file.write("Alpha is:{} \n".format(alpha))
    file.write("Training error: {}\n".format(tr_count / len(training_mails)))
    file.write("False Positives for Spam in training: {}\n".format(tr_false_assumptions[0] / nr_of_spam[0]))
    file.write("False Negatives for Spam in training: {}\n".format(tr_false_assumptions[1] / nr_of_spam[0]))
    file.write("Test error: {}\n".format(te_count / len(test_mails)))
    file.write("False Positives for Spam in test: {}\n".format(te_false_assumptions[0] / nr_of_spam[1]))
    file.write("False Negatives for Spam in test: {}\n".format(te_false_assumptions[1] / nr_of_spam[1]))

    alpha = 0.1

    tr_count = 0
    te_count = 0
    tr_false_assumptions = [0, 0]
    te_false_assumptions = [0, 0]

    for mail in training_mails:
        res = label_document(mail[1], ham_p, spam_p, ham_dict, spam_dict, h_count, s_count, lamda, alpha, v_num)
        if res > 0 and "spam" in mail[0]:
            tr_count += 1
        elif res <= 0 and "ham" in mail[0]:
            tr_count += 1
        elif res > 0 and "ham" in mail[0]:
            tr_false_assumptions[0] += 1
        elif res <=0 and "spam" in mail[0]:
            tr_false_assumptions[1] += 1

    for mail in test_mails:
        res = label_document(mail[1], ham_p, spam_p, ham_dict, spam_dict, h_count, s_count, lamda, alpha, v_num)
        if res > 0 and "spam" in mail[0]:
            te_count += 1
        elif res <= 0 and "ham" in mail[0]:
            te_count += 1
        elif res > 0 and "ham" in mail[0]:
            te_false_assumptions[0] += 1
        elif res <=0 and "spam" in mail[0]:
            te_false_assumptions[1] += 1

    file.write("Alpha is:{} \n".format(alpha))
    file.write("Training error: {}\n".format(tr_count / len(training_mails)))
    file.write("False Positives for Spam in training: {}\n".format(tr_false_assumptions[0] / nr_of_spam[0]))
    file.write("False Negatives for Spam in training: {}\n".format(tr_false_assumptions[1] / nr_of_spam[0]))
    file.write("Test error: {}\n".format(te_count / len(test_mails)))
    file.write("False Positives for Spam in test: {}\n".format(te_false_assumptions[0] / nr_of_spam[1]))
    file.write("False Negatives for Spam in test: {}\n".format(te_false_assumptions[1] / nr_of_spam[1]))

    alpha = 1

    tr_count = 0
    te_count = 0
    tr_false_assumptions = [0, 0]
    te_false_assumptions = [0, 0]

    for mail in training_mails:
        res = label_document(mail[1], ham_p, spam_p, ham_dict, spam_dict, h_count, s_count, lamda, alpha, v_num)
        if res > 0 and "spam" in mail[0]:
            tr_count += 1
        elif res <= 0 and "ham" in mail[0]:
            tr_count += 1
        elif res > 0 and "ham" in mail[0]:
            tr_false_assumptions[0] += 1
        elif res <=0 and "spam" in mail[0]:
            tr_false_assumptions[1] += 1

    for mail in test_mails:
        res = label_document(mail[1], ham_p, spam_p, ham_dict, spam_dict, h_count, s_count, lamda, alpha, v_num)
        if res > 0 and "spam" in mail[0]:
            te_count += 1
        elif res <= 0 and "ham" in mail[0]:
            te_count += 1
        elif res > 0 and "ham" in mail[0]:
            te_false_assumptions[0] += 1
        elif res <=0 and "spam" in mail[0]:
            te_false_assumptions[1] += 1

    file.write("Alpha is:{} \n".format(alpha))
    file.write("Training error: {}\n".format(tr_count / len(training_mails)))
    file.write("False Positives for Spam in training: {}\n".format(tr_false_assumptions[0] / nr_of_spam[0]))
    file.write("False Negatives for Spam in training: {}\n".format(tr_false_assumptions[1] / nr_of_spam[0]))
    file.write("Test error: {}\n".format(te_count / len(test_mails)))
    file.write("False Positives for Spam in test: {}\n".format(te_false_assumptions[0] / nr_of_spam[1]))
    file.write("False Negatives for Spam in test: {}\n".format(te_false_assumptions[1] / nr_of_spam[1]))

    file.close()


if __name__ == "__main__":
    # the model
    stop_words = get_stopwords()
    # documents already tokenized and mapped between labels and histograms
    training_mails = read_in_emails("train.txt", stop_words)
    test_mails = read_in_emails("test.txt", stop_words)
    # probability of ham and spam mails in training
    ham_p, spam_p = count_cat_of_mail(training_mails)
    # histograms for both classes
    ham_dict, spam_dict = dict_of_cat(training_mails)
    # count of words in each category
    h_count, s_count = count_of_words_in_cat(ham_dict, spam_dict)

    model_results(training_mails, test_mails, ham_p, spam_p, ham_dict, spam_dict, h_count, s_count)