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
    print num_rows
    curr_row = 0
    for curr_row in random_idx_list:
        if curr_row == 0:
            continue
        cell_value = worksheet.cell_value(curr_row, 0)
        followers = worksheet.cell_value(curr_row, 1)
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

def label_data_jaccard(tweets, rumor_word_count, threshold,rumor_list, th, followerDict):
    labeled_tweets  = {k : [] for k in range(rumor_list.__len__()+1)}
    # stemmer = PorterStemmer()
    for tweet in tweets:
        tweet_position = 11
        category = None
        tweet_words_set = set(sorted(tweet.split(), key = lambda x:x))
        new_words_set = set()
        jaccard_coeff = 0
        for idx, key in enumerate(rumor_word_count):
            rumor_word_dict = rumor_word_count[key]

            top_keys = sorted(rumor_word_dict, key = rumor_word_dict.get, reverse=True)[:int(th)]
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
        if tweet_position != 11 and int(followerDict[tweet]) > 50000:
            rumor_word_dict = rumor_word_count[category]
            for word in tweet_words_set:
                if word in rumor_word_dict:
                    rumor_word_dict[word] += 1

            for word in new_words_set:
                if word not in rumor_word_dict:
                    rumor_word_dict[word] = 0

            rumor_word_count[category] = rumor_word_dict

    return labeled_tweets, rumor_word_count


def label_data_graph(tweets, rumor_edge_count, threshold, len, th, followerDict):
    labeled_tweets  = {k : [] for k in range(len+1)}
    if tweets.__len__() == 0:
        return labeled_tweets, rumor_edge_count

    for tweet in tweets:
        if tweet.strip().__len__() == 0:
            continue

        tweet_position = 11
        category = None
        tweet_edge_set = set(sorted(tuples(tweet.split()).keys(), key = lambda x:x))
        new_words_set = set()
        jaccard_coeff = 0
        for idx, key in enumerate(rumor_edge_count):

            rumor_edge_dict = rumor_edge_count[key]

            top_keys = sorted(rumor_edge_dict, key = rumor_edge_dict.get, reverse=True)[:int(th)]
            new_rumour_dict = {}
            for top in top_keys:
                new_rumour_dict[top] = rumor_edge_dict[top]

            rumor_edge_set = set(sorted(new_rumour_dict.keys(), key = lambda x:x))

            intersection_set = rumor_edge_set.intersection(tweet_edge_set)
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
        if tweet_position != 11 and int(followerDict[tweet]) > 50000:
            rumor_edge_dict = rumor_edge_count[category]
            for word in tweet_edge_set:
                if word in rumor_edge_dict:
                    rumor_edge_dict[word] += 1

            for word in new_words_set:
                if word not in rumor_edge_dict:
                    rumor_edge_dict[word] = 0

            rumor_edge_count[category] = rumor_edge_dict

    return labeled_tweets, rumor_edge_count