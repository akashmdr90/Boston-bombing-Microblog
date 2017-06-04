# coding: utf-8
__author__ = 'sherlock'
import numpy as np
import SETTINGS
import cPickle
import re
import matplotlib.pyplot as plt
import random
from nltk.corpus import wordnet as wn
from nltk.stem.porter import *
import os
import networkx as nx
import xlrd

def readConceptDictionary():
    stemmer = PorterStemmer()
    workbook = xlrd.open_workbook("Words_Dictionary_Rohit_Revised.xls")
    worksheet = workbook.sheet_by_name('Sheet1')
    num_rows = worksheet.nrows
    conceptDict = {}
    curr_row = 0
    for curr_row in range(num_rows):
        if curr_row == 0:
            continue
        # concept =  worksheet.cell_value(curr_row, 0).encode('utf-8').lower()
        # concept = str(stemmer.stem(concept))
        word =  worksheet.cell_value(curr_row, 0).encode('utf-8').lower()
        word = str(stemmer.stem(word))
        concept = worksheet.cell_value(curr_row, 1).encode('utf-8').lower()
        if concept is "":
            continue
        # word = str(stemmer.stem(word))
        conceptDict[word] = concept
        # if concept not in conceptDict:
        #     conceptDict[concept] = concept
        curr_row += 1

    return conceptDict

def tweet_from_source(workbook, stop_words):

    originalTweets = {}
    tweetLabel = {}
    tweets = []
    mentions = re.compile(r'((?:\@|https?\://)\S+)')
    urls = re.compile(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')
    dots = re.compile(r'[\.\!\$\?\*\@\#\%\&\"\'\“\\\/\[\]\{\}\;\:\<\,\>\.\|\`\~]+')
    non_ascii = re.compile(r'[^\x00-\x7F]+')

    worksheet = workbook.sheet_by_name('Confusion Matrix')
    num_rows = worksheet.nrows
    random_idx_list = random.sample(range(num_rows), num_rows)
    curr_row = 0
    for curr_row in random_idx_list:
        if curr_row == 0:
            continue
        cell_value = worksheet.cell_value(curr_row, 1)
        if cell_value != "":
            raw_tweet = cell_value.encode('utf-8')
            tweet_without_urls = re.sub(urls,' ', raw_tweet)
            tweet_without_mentions = re.sub(mentions,' ',tweet_without_urls)
            tweet_without_end_dots = re.sub(dots,' ',tweet_without_mentions)
            tweet_without_non_ascii = re.sub(non_ascii, '', tweet_without_end_dots)
            tweet_without_stop_words = " ".join(word for word in tweet_without_non_ascii.split() if word.lower() not in stop_words)

            if "detonations" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'detonations',"detonation",tweet_without_stop_words)

            if "libraries" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'detonations',"library",tweet_without_stop_words)

            if "no-fly" in tweet_without_stop_words or "nofly" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'no-fly|nofly',"no fly",tweet_without_stop_words)

            if "nope" in tweet_without_stop_words or "none" in tweet_without_stop_words or "not" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'nope|none|not',"no",tweet_without_stop_words)

            if "12" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'12',"twelve",tweet_without_stop_words)

            if "guard" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'guarded|guarding',"guard",tweet_without_stop_words)

            tweets.append(tweet_without_stop_words.lower())
            tweetLabel[tweet_without_stop_words.lower()] = worksheet.cell_value(curr_row, 2)
            originalTweets[tweet_without_stop_words.lower()] = raw_tweet

        # curr_row += 1

    return tweets, num_rows, tweetLabel

def tweet_from_source_with_Followers(workbook, stop_words):

    originalTweets = {}
    tweetLabel = {}
    tweets = []
    followerDict = {}
    mentions = re.compile(r'((?:\@|https?\://)\S+)')
    urls = re.compile(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')
    dots = re.compile(r'[\.\!\$\?\*\@\#\%\&\"\'\“\\\/\[\]\{\}\;\:\<\,\>\.\|\`\~]+')
    non_ascii = re.compile(r'[^\x00-\x7F]+')

    worksheet = workbook.sheet_by_name('Followers')
    num_rows = worksheet.nrows
    random_idx_list = random.sample(range(num_rows), num_rows)
    curr_row = 0
    for curr_row in random_idx_list:
        if curr_row == 0:
            continue
        cell_value = worksheet.cell_value(curr_row, 0)
        followers = worksheet.cell_value(curr_row, 3)
        if cell_value != "":
            raw_tweet = cell_value.encode('utf-8')
            tweet_without_urls = re.sub(urls,' ', raw_tweet)
            tweet_without_mentions = re.sub(mentions,' ',tweet_without_urls)
            tweet_without_end_dots = re.sub(dots,' ',tweet_without_mentions)
            tweet_without_non_ascii = re.sub(non_ascii, '', tweet_without_end_dots)
            tweet_without_stop_words = " ".join(word for word in tweet_without_non_ascii.split() if word.lower() not in stop_words)

            if "detonations" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'detonations',"detonation",tweet_without_stop_words)

            if "libraries" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'detonations',"library",tweet_without_stop_words)

            if "no-fly" in tweet_without_stop_words or "nofly" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'no-fly|nofly',"no fly",tweet_without_stop_words)

            if "nope" in tweet_without_stop_words or "none" in tweet_without_stop_words or "not" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'nope|none|not',"no",tweet_without_stop_words)

            if "12" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'12',"twelve",tweet_without_stop_words)

            if "guard" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'guarded|guarding',"guard",tweet_without_stop_words)

            tweets.append(tweet_without_stop_words.lower())
            tweetLabel[tweet_without_stop_words.lower()] = worksheet.cell_value(curr_row, 2)
            originalTweets[tweet_without_stop_words.lower()] = raw_tweet

            followerDict[tweet_without_stop_words.lower()] = followers

    return tweets, num_rows, tweetLabel, followerDict


def get_rumor_list(wb, category, stop_words):
    f = open(SETTINGS.stop_words_file)
    worksheet = wb.sheet_by_name('Rumor Examples')
    num_rows = worksheet.nrows - 1
    curr_row = 0
    clean_rumor_list = []
    if category == 'Description':
        num = 2
    else:
        num = 1

    while curr_row < (num_rows):
        curr_row += 1
        cell_value = worksheet.cell_value(curr_row, num)
        text = " ".join(word for word in cell_value.split() if word.lower() not in stop_words)
        text = re.sub(r',\"\':;<>\?',"",text)
        clean_rumor_list.append(text.encode('utf-8').lower())

    return clean_rumor_list


def label_data(tweets, rumor_word_count, threshold,rumor_list, wordnetDict):
    labeled_tweets  = {k : [] for k in range(rumor_list.__len__()+1)}
    stemmer = PorterStemmer()
    for tweet in tweets:
        tweet_position = 11
        category = None
        # tweet_words_set = set(sorted(tweet.split(), key = lambda x:x))
        tweet_words_set = set()
        for token in tweet.split():
            try:
                token = str(stemmer.stem(token))
            except Exception as e:
                continue
            if token in wordnetDict:
                syn = wordnetDict[token]
                if type(syn) is str:
                    tweet_words_set.add(syn)
                # else:
                #     for s in syn:
                #         tokens = s.lemma_names()
                #         for tok in tokens:
                #             tweet_words_set.add(tok)

            tweet_words_set.add(token)

        new_words_set = set()
        jaccard_coeff = 0
        for idx, key in enumerate(rumor_word_count):
            rumor_word_dict = rumor_word_count[key]

            top_keys = sorted(rumor_word_dict, key = rumor_word_dict.get, reverse=True)[:10]
            new_rumour_dict = {}
            for top in top_keys:
                new_rumour_dict[top] = rumor_word_dict[top]
            rumor_words_set = set(sorted(new_rumour_dict.keys(), key = lambda x:x))

            intersection_set = rumor_words_set.intersection(tweet_words_set)
            union_set = rumor_words_set.union(tweet_words_set)
            jaccard_coeff_temp = max(jaccard_coeff, len(intersection_set) / float(len(union_set)))
            difference = tweet_words_set.difference(rumor_words_set)

            if(jaccard_coeff_temp != jaccard_coeff):
                tweet_position = idx
                category = key
                new_words_set = difference

            jaccard_coeff = jaccard_coeff_temp


#Checking if it is greater than threshold, then adding that tweet to appropriate position
        if jaccard_coeff <= threshold:
            labeled_tweets[11].append(tweet)
        else:
            labeled_tweets[tweet_position].append(tweet)

#Now we will update the rumor dictionary, but only if it is not NA and only for classified rumor
        if tweet_position != 11:
            rumor_word_dict = rumor_word_count[category]
            for word in tweet_words_set:
                if word in rumor_word_dict:
                    rumor_word_dict[word] += 1
                if word in wordnetDict:
                    parent = wordnetDict[word]
                    if parent in rumor_word_dict:
                        rumor_word_dict[parent] += 1

            for word in new_words_set:
                if word in wordnetDict:
                    parent = wordnetDict[word]
                    if parent not in rumor_word_dict:
                        rumor_word_dict[parent] = 0

                else:
                    if word not in rumor_word_dict:
                        syns = wn.synsets(word)
                        for s in syns:
                            tokens = s.lemma_names()
                            for token in tokens:
                                token = str(stemmer.stem(token))
                                if token not in wordnetDict:
                                    wordnetDict[token] = word
                        rumor_word_dict[word] = 0

            rumor_word_count[category] = rumor_word_dict

    return labeled_tweets, rumor_word_count, wordnetDict

def labelDataJaccardConcept(tweets, rumor_word_count, threshold,rumor_list, conceptDict, th, followerDict):
    labeled_tweets  = {k : [] for k in range(rumor_list.__len__()+1)}
    # stemmer = PorterStemmer()
    for tweet in tweets:
        tweet_position = 11
        category = None
        # tweet_words_set = set(sorted(tweet.split(), key = lambda x:x))
        tweet_words_set = set()
        tweetBody = {}
        for token in tweet.split():
            # try:
            #     token = str(stemmer.stem(token))
            # except Exception as e:
            #     continue
            concept = token
            if token in conceptDict:
                concept = conceptDict[token]
                if concept not in tweetBody:
                    tweetBody[concept] = [token]
                else:
                    li = tweetBody[concept]
                    li.append(token)
                    tweetBody[concept] = li

            tweet_words_set.add(concept)

        new_words_set = set()
        jaccard_coeff = 0
        for idx, key in enumerate(rumor_word_count):
            rumor_word_dict = rumor_word_count[key]

            top_keys = sorted(rumor_word_dict, key = rumor_word_dict.get, reverse=True)[:int(th)]
            new_rumour_dict = {}
            for top in top_keys:
                new_rumour_dict[top] = rumor_word_dict[top]

            rumor_words_set = set()
            for tok in new_rumour_dict.keys():
                if tok in conceptDict:
                    tok = conceptDict[tok]
                rumor_words_set.add(tok)
            # rumor_words_set = set(sorted(new_rumour_dict.keys(), key = lambda x:x))

            intersection_set = rumor_words_set.intersection(tweet_words_set)
            union_set = rumor_words_set.union(tweet_words_set)
            jaccard_coeff_temp = max(jaccard_coeff, len(intersection_set) / float(len(union_set)))
            difference = tweet_words_set.difference(rumor_words_set)

            if(jaccard_coeff_temp != jaccard_coeff):
                tweet_position = idx
                category = key
                new_words_set = set()
                for word in difference:
                    if word in tweetBody:
                        li = tweetBody[word]
                        for t in li:
                            new_words_set.add(t)
                    else:
                        new_words_set.add(word)
                # new_words_set = difference

            jaccard_coeff = jaccard_coeff_temp


#Checking if it is greater than threshold, then adding that tweet to appropriate position
        if jaccard_coeff > threshold:
            labeled_tweets[tweet_position].append(tweet)
        else:
            labeled_tweets[11].append(tweet)

#Now we will update the rumor dictionary, but only if it is not NA and only for classified rumor
        if tweet_position != 11 and int(followerDict[tweet]) > 1000:
            rumor_word_dict = rumor_word_count[category]
            for word in tweet_words_set:
                if word in rumor_word_dict:
                    rumor_word_dict[word] += 1

            for word in new_words_set:
                rumor_word_dict[word] = 0

            rumor_word_count[category] = rumor_word_dict

    return labeled_tweets, rumor_word_count


def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

def plot_Roc_curves(x_roc_values,y_roc_values, rumor_category, type, thresh, style):
     for key in rumor_category:
        x = x_roc_values[key]
        y = y_roc_values[key]
        plt.figure()
        plt.plot(x,y,marker = "o")
        plt.xlabel('1 - Specificity (False Positive Rate)')
        plt.ylabel('Sensitivity (True Positive Rate)')
        plt.title('ROC for '+key)
        plt.grid(True)
        plt.savefig(os.path.dirname(os.path.realpath(__file__))+"/Images/"+style+"/"+thresh+"/"+type+"_"+key+"_1.png", bbox_inches='tight')

def plot_good_ROC(rumor_category, type, thresh, style):
    x_roc_values = cPickle.load(open("x_roc_values_Graph"+type+".p", "rb"))
    y_roc_values = cPickle.load(open("y_roc_values_Graph"+type+".p", "rb"))
    for key in rumor_category:
        x = x_roc_values[key]
        y = y_roc_values[key]

        fig, ax = preparePlot(np.arange(0., 1.1, 0.1), np.arange(0., 1.1, 0.1))
        ax.set_xlim(-.05, 1.05), ax.set_ylim(-.05, 1.05)
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        plt.plot(x, y, color='#8cbfd0', linestyle='-', linewidth=3.)
        plt.plot((0., 1.), (0., 1.), linestyle='--', color='#d6ebf2', linewidth=2.)  # Baseline model
        plt.savefig(os.path.dirname(os.path.realpath(__file__))+"/Images/"+style+"/"+thresh+"/"+type+"_"+key+"_2.png", bbox_inches='tight')

def calculations(sorted_scores, rumor_category, type, thresh, style):
    x_roc_values = {}
    y_roc_values = {}

    # sorted_scores = sorted(scores.items(), key = operator.itemgetter(0))
    for score, tptnlist in sorted_scores:
        tp1 = tptnlist[0]
        fp1 = tptnlist[1]
        fn1 = tptnlist[2]
        tn1 = tptnlist[3]
        for key in rumor_category:
            specificity = float(tn1.get(key,0))/(tn1.get(key,0) + fp1.get(key,0))
            sensitivity = float(tp1.get(key,0))/(tp1.get(key,0) + fn1.get(key,0))
            if key not in x_roc_values:
                x_roc_values[key] = []
                y_roc_values[key] = []
            x_roc_values[key].append(1-specificity)
            y_roc_values[key].append(sensitivity)
    cPickle.dump(x_roc_values, open("x_roc_values_Graph"+type+".p", "wb"))
    cPickle.dump(y_roc_values, open("y_roc_values_Graph"+type+".p", "wb"))

    # x_roc_values = cPickle.load(open('x_roc_values.p', 'rb'))
    # y_roc_values = cPickle.load(open('y_roc_values.p', 'rb'))
    plot_Roc_curves(x_roc_values,y_roc_values,rumor_category, type, thresh, style)
    plot_good_ROC(rumor_category,type, thresh, style)


def construct_graph(rumor, wordnetDict):
    word_array = rumor.split()
    stemmer = PorterStemmer()
    previous = str(stemmer.stem(word_array[0]))
    if previous not in wordnetDict:
        p_syns = wn.synsets(previous)
        if len(p_syns) == 0:
            p_syns = wn.synsets(word_array[0])
            if len(p_syns) == 0:
                p_syns.append(word_array[0])
            wordnetDict[word_array[0]] = p_syns

        wordnetDict[previous] = p_syns
    else:
        p_syns = wordnetDict[previous]

    edge_dict = {}
    for i in range(1,len(word_array)):
        # word_array[i] = re.sub(non_ascii, '', word_array[i])
        # current = str(stemmer.stem(str(word_array[i])))
        current = str(stemmer.stem(word_array[i]))
        if current not in wordnetDict:
            c_syns = wn.synsets(current)
            if len(c_syns) == 0:
                c_syns = wn.synsets(word_array[i])
                if len(c_syns) == 0:
                    c_syns.append(word_array[i])
                wordnetDict[word_array[i]] = c_syns

            wordnetDict[current] = c_syns
        else:
            c_syns = wordnetDict[current]

        for p in p_syns:

            if type(p) is str:
                p_tokens = [p]
            else:
                p_tokens = p.lemma_names()

            for c in c_syns:
                if type(c) is str:
                    c_tokens = [c]
                else:
                    c_tokens = c.lemma_names()

                prevTokSet = set()
                for ptok in p_tokens:
                    if ptok in prevTokSet or "_" in ptok:
                        continue
                    prevTokSet.add(ptok)
                    # ptok = str(stemmer.stem(str(re.sub(non_ascii, '', ptok))))
                    ptok = str(stemmer.stem(ptok))
                    currTokSet = set()
                    for ctok in c_tokens:
                        if ctok in currTokSet or "_" in ctok:
                            continue
                        currTokSet.add(ctok)
                        # ctok = str(stemmer.stem(str(re.sub(non_ascii, '', ctok))))
                        ctok = str(stemmer.stem(ctok))
                        if (ptok,ctok) not in edge_dict:
                            edge = (ptok,ctok)
                            # if edge not in edge_dict:
                            edge_dict[edge] = 0

        p_syns = c_syns


    return edge_dict, wordnetDict

def create_tuples(word_array):
    edgeSet = set()
    stemmer = PorterStemmer()
    previous = str(stemmer.stem(word_array[0]))
    for i in range(1,len(word_array)):
        current = str(stemmer.stem(word_array[i]))
        edgeSet.add((previous, current))
        previous = current

    return edgeSet

def tuples(word_array):
    edgeDict = dict()
    # stemmer = PorterStemmer()
    # previous = str(stemmer.stem(word_array[0]))
    previous = word_array[0]
    for i in range(1,len(word_array)):
        # current = str(stemmer.stem(word_array[i]))
        current = word_array[i]
        edgeDict[(previous, current)] = 0
        previous = current

    return edgeDict

def tuplesWithConcept(word_array, conceptDict):
    edgeDict = dict()
    stemmer = PorterStemmer()
    previous = str(stemmer.stem(word_array[0]))
    previous = word_array[0]
    concept1 = previous
    if previous in conceptDict:
        concept1 = conceptDict[previous]
    for i in range(1,len(word_array)):
        # current = str(stemmer.stem(word_array[i]))
        current = word_array[i]
        concept2 = current
        if current in conceptDict:
            concept2 = conceptDict[current]
        edgeDict[(concept1, concept2)] = 0
        edgeDict[(concept2, concept1)] = 0
        concept1 = concept2

    return edgeDict

def label_data_graph(tweets, rumor_edge_count, threshold, len, wordnetDict):
    labeled_tweets  = {k : [] for k in range(len+1)}
    stemmer = PorterStemmer()
    num = 0
    for tweet in tweets:
        num += 1
        print "Reading Tweet "+str(num)
        if tweet.strip().__len__() == 0:
            continue
        k = 0
        tweet_position = 11
        category = None
        # tweet_graph, wordnetDict = construct_graph(tweet, wordnetDict)
        # tweetSet = create_tuples(tweet.split())
        tweet_edge_set = create_tuples(tweet.split()) #set(sorted(create_tuples(tweet.split()).keys(), key = lambda x:x))
        new_words_set = set()
        jaccard_coeff = 0
        for idx, key in enumerate(rumor_edge_count):
            rumor_edge_dict = rumor_edge_count[key]
            # if num < 10:
            #     rumor_edge_set = set(rumor_edge_dict.keys())
            # else:
            top_keys = sorted(rumor_edge_dict, key = rumor_edge_dict.get, reverse=True)[:25]
            new_rumour_dict = {}
            for top in top_keys:
                new_rumour_dict[top] = rumor_edge_dict[top]

            rumor_edge_set = set(sorted(new_rumour_dict.keys(), key = lambda x:x))

            intersection_set = rumor_edge_set.intersection(tweet_edge_set)
            # union_set = rumor_edge_set.union(tweet_edge_set)
            if rumor_edge_set.__len__() == 0:
                jaccard_coeff_temp
            else:
                jaccard_coeff_temp = max(jaccard_coeff, intersection_set.__len__() / float(rumor_edge_set.__len__()))
            difference = tweet_edge_set.difference(rumor_edge_set)

            if(jaccard_coeff_temp != jaccard_coeff):
                tweet_position = idx
                category = key
                new_words_set = difference
            jaccard_coeff = jaccard_coeff_temp


#Checking if it is greater than threshold, then adding that tweet to appropriate position
        if jaccard_coeff <= threshold:
            labeled_tweets[11].append(tweet)
        else:
            labeled_tweets[tweet_position].append(tweet)

#No we will update the rumor dictionary, but only if it is not NA and only for classified rumor
        if tweet_position != 11:

            rumor_edge_dict = rumor_edge_count[category]
            for word in tweet_edge_set:
                if word in rumor_edge_dict:
                    rumor_edge_dict[word] += 1

            for word in new_words_set:
                if word not in rumor_edge_dict:
                    rumor_edge_dict[word] = 0

            rumor_edge_count[category] = rumor_edge_dict

    return labeled_tweets, rumor_edge_count, wordnetDict

def labelGraphConcept(tweets, rumor_edge_count, threshold, len, conceptDict, th, followerDict):
    labeled_tweets  = {k : [] for k in range(len+1)}
    if tweets.__len__() == 0:
        return labeled_tweets, rumor_edge_count
    stemmer = PorterStemmer()
    num = 0
    for tweet in tweets:
        num += 1
        # print "Reading Tweet "+str(num)
        if tweet.strip().__len__() == 0:
            continue

        tweetBody = {}
        for token in tweet.split():
            concept = token
            if token in conceptDict:
                concept = conceptDict[token]
                if concept not in tweetBody:
                    tweetBody[concept] = [token]
                else:
                    li = tweetBody[concept]
                    li.append(token)
                    tweetBody[concept] = li

        k = 0
        tweet_position = 11
        category = None
        # tweet_graph, wordnetDict = construct_graph(tweet, wordnetDict)
        # tweetSet = create_tuples(tweet.split())
        tweet_edge_set = set(tuplesWithConcept(tweet.split(), conceptDict).keys()) #set(sorted(create_tuples(tweet.split()).keys(), key = lambda x:x))
        new_words_set = set()
        jaccard_coeff = 0
        for idx, key in enumerate(rumor_edge_count):
            rumor_edge_dict = rumor_edge_count[key]
            # if num < 10:
            #     rumor_edge_set = set(rumor_edge_dict.keys())
            # else:
            top_keys = sorted(rumor_edge_dict, key = rumor_edge_dict.get, reverse=True)[:int(th)]
            new_rumour_dict = {}
            for top in top_keys:
                new_rumour_dict[top] = rumor_edge_dict[top]

            # rumor_edge_set = set(sorted(new_rumour_dict.keys(), key = lambda x:x))
            rumor_edge_set = set()
            for edge in new_rumour_dict:
                word1 = edge[0]
                word2 = edge[1]
                if word1 in conceptDict:
                    word1 = conceptDict[word1]
                if word2 in conceptDict:
                    word2 = conceptDict[word2]
                rumor_edge_set.add((word1,word2))
            intersection_set = rumor_edge_set.intersection(tweet_edge_set)
            # union_set = rumor_edge_set.union(tweet_edge_set)
            if rumor_edge_set.__len__() == 0:
                jaccard_coeff_temp = 0
            else:
                jaccard_coeff_temp = max(jaccard_coeff, intersection_set.__len__() / float(rumor_edge_set.__len__()))
            difference = tweet_edge_set.difference(rumor_edge_set)

            if(jaccard_coeff_temp != jaccard_coeff):
                tweet_position = idx
                category = key
                new_words_set = set()
                for tuple in difference:
                    word1 = tuple[0]
                    word2 = tuple[1]
                    if word1 in tweetBody:
                        li = tweetBody[word1]
                        word1 = li[0]
                    if word2 in tweetBody:
                        li = tweetBody[word2]
                        word2 = li[0]
                    new_words_set.add((word1,word2))

                # new_words_set = difference
            jaccard_coeff = jaccard_coeff_temp


#Checking if it is greater than threshold, then adding that tweet to appropriate position
        if jaccard_coeff > threshold:
            labeled_tweets[tweet_position].append(tweet)
        else:
            labeled_tweets[11].append(tweet)

#No we will update the rumor dictionary, but only if it is not NA and only for classified rumor
        if tweet_position != 11 and int(followerDict[tweet]) > 1000:

            rumor_edge_dict = rumor_edge_count[category]
            for word in tweet_edge_set:
                if word in rumor_edge_dict:
                    rumor_edge_dict[word] += 1

            for word in new_words_set:
                if word not in rumor_edge_dict:
                    rumor_edge_dict[word] = 0

            rumor_edge_count[category] = rumor_edge_dict


    return labeled_tweets, rumor_edge_count
