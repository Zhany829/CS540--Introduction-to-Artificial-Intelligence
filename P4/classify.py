import os;
import math;
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # TODO: add your code here
    file = open(filepath, "r",encoding='utf-8')
    count_none = 0
    for line in file:
        line = line.rstrip('\n')
        if line in vocab:
            if line not in bow:
                bow.update({line: 1})
            else:
                bow[line] += 1
        else:
            count_none += 1

    if(count_none != 0):
        bow.update({None: count_none})
    return bow

def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    dataset = []
    # TODO: add your code here
    for folder in os.listdir(directory):
         year = folder
         for file in os.listdir(directory + folder):
            bow = create_bow(vocab, directory + folder + "/" + file)
            dataset.append({"label": year, "bow": bow})

    return dataset

def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """
    vocab = []
    # TODO: add your code here
    all_vocab = []
    # open file under the training_directory and analyze
    for folder in os.listdir(directory):
        for filename in os.listdir(directory + "/"+folder):
            f = open(directory +"/"+ folder + "/" + filename, "r",encoding='utf-8')
            for line in f:
                line = line.rstrip('\n')
                flag = 0
                for voca in all_vocab:
                    if voca["word"] == line:
                        voca["count"] += 1
                        flag = 1
                if flag == 0:
                    new_type = {
                        "word": line,
                        "count": 1
                    }
                    all_vocab.append(new_type)
                    
    for word_type in all_vocab:
        if word_type["count"] >= cutoff:
            vocab.append(word_type["word"])

    return sorted(vocab)

def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """
    smooth = 1 # smoothing factor
    logprob = {}
    # TODO: add your code here
    for label in label_list:
        logprob.update({label: 0})
    for train_dict in training_data:
        logprob[train_dict["label"]] += smooth
    for label in logprob:
        logprob[label] = math.log((logprob[label]+smooth)/(len(training_data)+2*smooth))
    return logprob

def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """
    
    smooth = 1 # smoothing factor
    word_prob = {}
    # TODO: add your code here
    #update the count of given words
    for word in vocab:
        word_prob.update({word:smooth})
    word_prob.update({None:smooth})
    for item in training_data:
        if item["label"] == label:
            for token in item["bow"]:
                count = item["bow"][token]
                if token in word_prob.keys():
                    word_prob[token] += count
                else:
                    word_prob[None] += count
        else:
            continue
    total_count = 0
    for word in word_prob:
        total_count += word_prob[word]
    for word in word_prob:
        word_prob[word] = math.log(word_prob[word]/total_count)

    return word_prob

    
##################################################################################
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    # TODO: add your code here
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    log_prior = prior(training_data, ["2016", "2020"])
    log_2016 = p_word_given_label(vocab, training_data, "2016")
    log_2020 = p_word_given_label(vocab, training_data, "2020")
    retval.update({'vocabulary': vocab}) 
    retval.update({'log prior': log_prior})
    retval.update({'log p(w|y=2016)' : log_2016})
    retval.update({'log p(w|y=2020)' : log_2020})
    return retval


def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>, 
             'log p(y=2016|x)': <log probability of 2016 label for the document>, 
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    data_2016 = model['log prior']["2016"]
    data_2020 = model['log prior']["2020"]
    bow = create_bow(model["vocabulary"], filepath)
    for word in bow:
        for vocab in model['log p(w|y=2016)']:
            if vocab == word:
                data_2016 += bow[word] * model['log p(w|y=2016)'][vocab]
                data_2020 += bow[word] * model['log p(w|y=2020)'][vocab]
    year = 0
    if(data_2016 <= data_2020):
        year = 2020
    else:
        year = 2016
    retval.update({'log p(y=2020|x)': data_2020})
    retval.update({'log p(y=2016|x)': data_2016})
    retval.update({'predicted y': year})
    return retval


