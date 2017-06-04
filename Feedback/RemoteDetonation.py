# coding: utf-8
__author__ = 'sherlock'
import xlrd
import xlwt
import numpy as np
import SETTINGS
import cPickle
import re
import matplotlib.pyplot as plt
import time
import os
import operator
import random
from string import digits
from GraphFeedback import create_tuples as ct

def readConceptDictionary():
    workbook = xlrd.open_workbook("ConceptMap.xlsx")
    worksheet = workbook.sheet_by_name('Sheet1')
    num_rows = worksheet.nrows
    conceptDict = {}
    curr_row = 0
    for curr_row in range(num_rows):
        if curr_row == 0:
            continue
        concept =  worksheet.cell_value(curr_row, 0).encode('utf-8').lower()
        word = worksheet.cell_value(curr_row, 1).encode('utf-8').lower()
        conceptDict[word] = concept
        curr_row += 1

    return conceptDict


def tweet_from_source(workbook, stop_words):

    originalTweets = {}
    tweetLabel = {}
    tweets = []
    mentions = re.compile(r'((?:\@|https?\://)\S+)')
    urls = re.compile(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')
    dots = re.compile(r'[\.\!\$\?\*\@\#\%\&\"\'\“\\\/\[\]\{\}\;\:\<\,\>\.\|\`\~]+')

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
            tweet_without_urls = re.sub(urls,' ', raw_tweet).lower()
            tweet_without_mentions = re.sub(mentions,' ',tweet_without_urls)
            tweet_without_end_dots = re.sub(dots,' ',tweet_without_mentions)
            tweet_without_stop_words = " ".join(word for word in tweet_without_end_dots.split() if word.lower() not in stop_words)
            if "detonations" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'detonations',"detonation",tweet_without_stop_words)
            if "libraries" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'detonations',"library",tweet_without_stop_words)
            # if "no-fly" in tweet_without_stop_words or "nofly" in tweet_without_stop_words:
            #     tweet_without_stop_words = re.sub(r'no-fly|nofly',"no fly",tweet_without_stop_words)
            # if "flies" in tweet_without_stop_words:
            #     tweet_without_stop_words = re.sub(r'flies',"fly",tweet_without_stop_words)
            # if "nope" in tweet_without_stop_words or "none" in tweet_without_stop_words or "not" in tweet_without_stop_words:
            #     tweet_without_stop_words = re.sub(r'nope|none|not',"no",tweet_without_stop_words)
            if "guard" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'guarded|guarding',"guard",tweet_without_stop_words)
            if "12" in tweet_without_stop_words:
                tweet_without_stop_words = re.sub(r'12',"twelve",tweet_without_stop_words)
            tweet_without_stop_words = tweet_without_stop_words.translate(None,digits)
            tweets.append(tweet_without_stop_words.lower())
            tweetLabel[tweet_without_stop_words.lower()] = worksheet.cell_value(curr_row, 2)
            originalTweets[tweet_without_stop_words.lower()] = raw_tweet

        # curr_row += 1

    return tweets, num_rows, tweetLabel

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
        clean_rumor_list.append(text.encode('utf-8').lower())

    return clean_rumor_list

def get_tweets(workbook, stop_words):
    originalTweets = {}
    tweetLabel = {}
    tweets = []
    mentions = re.compile(r'((?:\@|https?\://)\S+)')
    urls = re.compile(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')
    dots = re.compile(r'[\.\!\$\?\*\@\#\%\&\"\'\“\\\/\[\]\{\}\;\:\<\,\>\.\|\`\~]+')

    worksheet = workbook.sheet_by_name('Confusion Matrix')
    num_rows = worksheet.nrows - 1
    curr_row = 0
    while curr_row < (num_rows):
        curr_row += 1
        cell_value = worksheet.cell_value(curr_row, 1)
        if cell_value != "":
            raw_tweet = cell_value.encode('utf-8')
            tweet_without_urls = re.sub(urls,' ', raw_tweet)
            tweet_without_mentions = re.sub(mentions,' ',tweet_without_urls)
            tweet_without_end_dots = re.sub(dots,' ',tweet_without_mentions)
            tweet_without_stop_words = " ".join(word for word in tweet_without_end_dots.split() if word.lower() not in stop_words)
            tweets.append(tweet_without_stop_words.lower())
            tweetLabel[tweet_without_stop_words.lower()] = worksheet.cell_value(curr_row, 2)
            originalTweets[tweet_without_stop_words.lower()] = raw_tweet

    return tweets, num_rows, tweetLabel

def label_data(tweets, rumor_word_count, threshold):
    labeled_tweets  = {k : [] for k in range(rumor_list.__len__()+1)}

    for tweet in tweets:
        tweet_position = 1
        category = None
        tweet_words_set = set(sorted(tweet.split(), key = lambda x:x))
        new_words_set = set()
        jaccard_coeff = 0
        if "remot" in tweet and "detonat" in tweet:
        # if "jfk" in tweet and "librar" in tweet and "fire" in tweet:
        # if "no" in tweet and "fly" in tweet and "zone" in tweet: #
        # if "saud" in tweet and "natio" in tweet:
        # if "twelve" in tweet and "dead" in tweet:
            for idx, key in enumerate(rumor_word_count):
                rumor_word_dict = rumor_word_count[key]

                top_keys = sorted(rumor_word_dict, key = rumor_word_dict.get, reverse=True)[:25]
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
            labeled_tweets[1].append(tweet)
        else:
            # if "no-fly" in tweet or "nofly" in tweet:
            #     tweet = re.sub(r'no-fly|nofly',"no fly",tweet)
            labeled_tweets[tweet_position].append(tweet)
            # print tweet

#Now we will update the rumor dictionary, but only if it is not NA and only for classified rumor
        if tweet_position != 1:
            rumor_word_dict = rumor_word_count[category]
            for word in tweet_words_set:
                if word in rumor_word_dict:
                    rumor_word_dict[word] += 1

            for word in new_words_set:
                if word not in rumor_word_dict:
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

def plot_Roc_curves(x_roc_values,y_roc_values, rumor_category, type):
     for key in rumor_category:
        x = x_roc_values[key]
        y = y_roc_values[key]
        plt.figure()
        plt.plot(x,y,marker = "o")
        plt.xlabel('1 - Specificity (False Positive Rate)')
        plt.ylabel('Sensitivity (True Positive Rate)')
        plt.title('ROC for '+key)
        plt.grid(True)
        # plt.savefig(SETTINGS.jaccardOut+"Images/Feedback/Combined/"+type+"/Combined_"+ type+"_"+key+"_1.png", bbox_inches='tight')
        plt.savefig(os.path.dirname(os.path.realpath(__file__))+"/Images/Combined/"+key+"_1.png", bbox_inches='tight')

def plot_good_ROC(rumor_category, type):
    x_roc_values = cPickle.load(open("x_roc_values_Jacc"+type+".p", "rb"))
    y_roc_values = cPickle.load(open("y_roc_values_Jacc"+type+".p", "rb"))
    for key in rumor_category:
        x = x_roc_values[key]
        y = y_roc_values[key]

        fig, ax = preparePlot(np.arange(0., 1.1, 0.1), np.arange(0., 1.1, 0.1))
        ax.set_xlim(-.05, 1.05), ax.set_ylim(-.05, 1.05)
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        plt.plot(x, y, color='#8cbfd0', linestyle='-', linewidth=3.)
        plt.plot((0., 1.), (0., 1.), linestyle='--', color='#d6ebf2', linewidth=2.)  # Baseline model
        # plt.savefig(SETTINGS.jaccardOut+"Images/Feedback/Combined/"+type+"/Combined_"+ type+"_"+key+"_2.png", bbox_inches='tight')
        plt.savefig(os.path.dirname(os.path.realpath(__file__))+"/Images/Combined/"+key+"_2.png", bbox_inches='tight')


def calculations(sorted_scores, rumor_category, type):
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
    cPickle.dump(x_roc_values, open("x_roc_values_Jacc"+type+".p", "wb"))
    cPickle.dump(y_roc_values, open("y_roc_values_Jacc"+type+".p", "wb"))

    # x_roc_values = cPickle.load(open('x_roc_values.p', 'rb'))
    # y_roc_values = cPickle.load(open('y_roc_values.p', 'rb'))
    plot_Roc_curves(x_roc_values,y_roc_values,rumor_category, type)
    plot_good_ROC(rumor_category,type)

def graph_label_data(tweets, rumor_edge_count, threshold):
    labeled_tweets  = {k : [] for k in range(rumor_list.__len__()+1)}

    for tweet in tweets:
        tweet_position = 1
        category = None
        tweet_edge_set = set(sorted(ct(tweet.split()).keys(), key = lambda x:x))
        new_words_set = set()
        jaccard_coeff = 0
        for idx, key in enumerate(rumor_edge_count):

            rumor_edge_dict = rumor_edge_count[key]

            top_keys = sorted(rumor_edge_dict, key = rumor_edge_dict.get, reverse=True)[:25]
            new_rumour_dict = {}
            for top in top_keys:
                new_rumour_dict[top] = rumor_edge_dict[top]

            rumor_edge_set = set(sorted(new_rumour_dict.keys(), key = lambda x:x))

            intersection_set = rumor_edge_set.intersection(tweet_edge_set)
            union_set = rumor_edge_set.union(tweet_edge_set)
            jaccard_coeff_temp = max(jaccard_coeff, len(intersection_set) / float(len(rumor_edge_set)))
            difference = tweet_edge_set.difference(rumor_edge_set)

            # if intersection_set.__len__() > 0:
            #     print 'mimanshu'
            if(jaccard_coeff_temp != jaccard_coeff):
                tweet_position = idx
                category = key
                new_words_set = difference
            jaccard_coeff = jaccard_coeff_temp

#Checking if it is greater than threshold, then adding that tweet to appropriate position
        if jaccard_coeff <= threshold:
            labeled_tweets[1].append(tweet)
        else:
            labeled_tweets[tweet_position].append(tweet)

#No we will update the rumor dictionary, but only if it is not NA and only for classified rumor
        if tweet_position != 1:
            rumor_edge_dict = rumor_edge_count[category]
            for word in tweet_edge_set:
                if word in rumor_edge_dict:
                    rumor_edge_dict[word] += 1

            for word in new_words_set:
                if word not in rumor_edge_dict:
                    rumor_edge_dict[word] = 0

            rumor_edge_count[category] = rumor_edge_dict

    return labeled_tweets, rumor_edge_count

if __name__ == "__main__":
    start_time = time.time()
    conceptDict = readConceptDictionary()
    workbook = xlrd.open_workbook(SETTINGS.dataset_folder+"dataset.xlsx")
    stop_words = open(SETTINGS.stop_words_file).read().splitlines()
    tweets, num_rows, tweetLabel = tweet_from_source(workbook, stop_words)
    rumor_category = {"NA","R1","All"}

    rumor_type = {'Description', 'Type'}
    for t in sorted(rumor_type):
        scores = {}
        # rumor_list = get_rumor_list(workbook,t, stop_words)
        # rumor_list = ['remote detonation']
        rumor_list = ['remote detonation']
        rumor_word_count = {i : {} for i in range(rumor_list.__len__())}
        rumor_edge_count = {i : {} for i in range(rumor_list.__len__())}
        k = 0
        for rumor in rumor_list:
            word_dict = {word : 0 for word in rumor.split()}
            word_array = rumor.split()
            rumor_word_count[k] = word_dict
            edge_dict = ct(word_array)
            rumor_edge_count[k] = edge_dict
            k += 1

        # threshold = np.arange(0,1,0.01)
        original_rumor_word_count = rumor_word_count
        original_rumor_edge_count = rumor_edge_count
        word_count_storage = {}
        threshold = [0.05]
        for val in threshold:
            i = 0
            rumor_edge_count = original_rumor_edge_count
            rumor_word_count = original_rumor_word_count
            jac_labeled_tweets, rumor_word_count = label_data(tweets, rumor_word_count, val)
            g_tweets = []
            na_tweets = []
            for key in jac_labeled_tweets:
                if key == 0: # or key == 2 or key == 3 or key == 4 or key == 6:
                    g_tweets += jac_labeled_tweets[key]
                else:
                    na_tweets += jac_labeled_tweets[key]

            labeled_tweets, rumor_edge_count = graph_label_data(g_tweets,rumor_edge_count,val)
            labeled_tweets[1] += na_tweets
            # word_count_storage[val] = rumor_word_count
            word_count_storage[val] = rumor_edge_count
            workbook_new = xlwt.Workbook()
            sheet = workbook_new.add_sheet("confusion Matrix")
            sheet2 = workbook_new.add_sheet("Word Count")
            sheet3 = workbook_new.add_sheet("Edge Count")
            sheet3.write(0,0,"Node1")
            sheet3.write(0,1,"Node2")
            sheet3.write(0,2,"Weight")
            wordC = rumor_word_count[0]
            edgeC = rumor_edge_count[0]
            rr = 1
            for word in wordC:
                wordx = re.sub(r'\W+', '', word)
                if (wordx != "" and len(wordx) > 2) or wordx == "no":
                    sheet2.write(rr,0,wordx.decode('ISO-8859-1'))
                    sheet2.write(rr,1,wordC[word]+1)
                    rr += 1

            rr = 1
            import networkx as nx
            import math
            G=nx.Graph()
            weight_map={}
            for edge in edgeC:
                word1 = edge[0]
                word2 = edge[1]
                word1 = re.sub(r'\W+', '', word1)
                word2 = re.sub(r'\W+', '', word2)
                if word1 == "no":
                    word1 = word1+" "
                if word2 == "no":
                    word2 = word2+" "
                if word1 != "" and word2 != "" and len(word1) > 2 and len(word2) > 2:
                    if "remotedetonation" in word1 or "remotedetonation" in word2:
                        continue
                    sheet3.write(rr,0,word1.decode('ISO-8859-1'))
                    sheet3.write(rr,1,word2.decode('ISO-8859-1'))
                    sheet3.write(rr,2,edgeC[edge]+1)
                    weight = edgeC[edge]+1
                    G.add_edge(word1, word2, weight = weight)

                    if weight in weight_map:
                        edge_list = weight_map[weight]
                        edge_list.append((word1,word2))
                        weight_map[weight]=edge_list
                    else:
                        edge_list=[(word1,word2)]
                        weight_map[weight]=edge_list
                    rr += 1
            # pos = nx.get_node_attributes(H,'pos')
            elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >3]
            esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=3]
            pos=nx.spring_layout(G)
            nx.draw_networkx_nodes(G,pos,node_size=700)
            nx.draw_networkx_edges(G,pos,edgelist=elarge,
                                width=2)
            nx.draw_networkx_edges(G,pos,edgelist=esmall,
                                width=1,alpha=0.5,edge_color='r',style='dashed')
            nx.draw_networkx_labels(G,pos,font_size=15, font_weight='bold', font_family='Helvetica')
            nx.draw_networkx_nodes(G,pos,node_color='w', font_size=15,font_family='sans-serif', node_size=1000, alpha=0.9, node_shape='s')
            plt.axis('off')
            plt.savefig("weighted_graph.png")
            # # edges
            # for key, value in weight_map.iteritems():
            #     nx.draw_networkx_edges(G,pos,edgelist=value,width=math.log(float(key),2))
            #     print key
            # nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif', )
            # labels = nx.get_edge_attributes(G,'weight')
            # nx.draw_networkx_edge_labels(G,edge_labels=labels)
            plt.show()
            sheet.write(0,0,"Tweets")
            sheet.write(0,1,"Category")
            sheet.write(0,2,"True Value")
            # workbook = xlrd.open_workbook("Files/"+type+"_out_"+i+".xls")
            # worksheet = workbook.sheet_by_name('Sheet1')
            # num_rows = worksheet.nrows - 1
            sheet2.write(0,0,"Word")
            sheet2.write(0,1,"Count")
            tp = {}
            tn = {}
            fp = {}
            fn = {}
            excel = {}
            for key in labeled_tweets:
                cat = None
                if key != 0 :#and key != 2 and key != 3 and key != 4 and key != 6:
                    cat = "NA"
                elif key == 0:
                    cat = "R1"
                # elif key == 2:
                #     cat = "R3"
                # elif key == 3:
                #     cat = "R4"
                # elif key == 4:
                #     cat = "R5"
                # elif key == 6:
                #     cat == "R7"

                value = labeled_tweets[key]
                for item in value:
                    if item in tweetLabel:
                        type = tweetLabel[item]
                        if type != "NA" and cat != "NA":
                            if "All" not in tp:
                                tp["All"] = 1
                            else:
                                tp["All"] = tp["All"] + 1

                        elif type == "NA" and cat != "NA":
                            if "All" not in fp:
                                fp["All"] = 1
                            else:
                                fp["All"] = fp["All"] + 1

                        elif type != "NA" and cat == "NA":
                            if "All" not in fn:
                                fn["All"] = 1
                            else:
                                fn["All"] = fn["All"] + 1

                        if type == cat:
                            if type not in tp:
                                tp[type] = 1
                            else:
                                tp[type] = tp[type] + 1

                        else:
                            if cat not in fp:
                                fp[cat] = 1
                            else:
                                fp[cat] = fp[cat] + 1
                            if type not in fn:
                                fn[type] = 1
                            else:
                                fn[type] = fn[type] + 1
                    excel[item] = cat

            for key in rumor_category:
                tn[key] = tweets.__len__() - tp.get(key,0) - fp.get(key,0) - fn.get(key,0)

            tptnlist = []
            tptnlist.append(tp)
            tptnlist.append(fp)
            tptnlist.append(fn)
            tptnlist.append(tn)
            scores[val] = tptnlist

            row = 1
            default = 'NA'
            print val
            for tweet in tweets:
                cc = excel.get(tweet)
                sheet.write(row,0,tweet.decode('ISO-8859-1'))#.decode('utf-8')
                sheet.write(row,1,cc)
                sheet.write(row,2,tweetLabel.get(tweet))
                row += 1
            save_path = os.getcwd()+"/Files/Jaccard/"+t
            workbook_new.save(save_path+"/out_"+str(val).encode('utf-8')+".xls")
            i += 1

        sorted_scores = sorted(scores.items(), key = operator.itemgetter(0))
        cPickle.dump(word_count_storage, open("word_count_storage_Graph_"+str(t)+".p", "wb"))
        cPickle.dump(scores, open("scores_Graph_"+t+".p", "wb"))
        # scores = cPickle.load(open("scores_Jacc"+t+".p", "rb"))
        calculations(sorted_scores,rumor_category,t)

        workbook_eval = xlwt.Workbook()
        sheet = workbook_eval.add_sheet("Evaluation")
        sheet.write(0,0,"Threshold")
        j = 0;
        for key in rumor_category:
            sheet.write(0,j+1,key+"_True Positive")
            sheet.write(0,j+2,key+"_False positive")
            sheet.write(0,j+3,key+"_True Negative")
            sheet.write(0,j+4,key+"_False Negative")
            sheet.write(0,j+5,key+"_Precision")
            sheet.write(0,j+6,key+"_Recall")
            sheet.write(0,j+7,key+"_F1-Measure")
            sheet.write(0,j+8,key+"_Accuracy")
            j += 8

        row = 1

        for score, tptnlist in sorted_scores:
            sheet.write(row,0,score)
            # tptnlist = sorted_scores[val];
            tp1 = tptnlist[0]
            fp1 = tptnlist[1]
            fn1 = tptnlist[2]
            tn1 = tptnlist[3]
            j = 0
            for key in rumor_category:
                p_1 = tp1.get(key,0) + fp1.get(key,0)
                precision = 0
                if p_1 != 0:
                    precision = float(tp1.get(key,0))/(p_1)
                recall = float(tp1.get(key,0))/(tp1.get(key,0) + fn1.get(key,0))
                sheet.write(row,j+1,tp1.get(key,0))
                sheet.write(row,j+2,fp1.get(key,0))
                sheet.write(row,j+3,tn1.get(key,0))
                sheet.write(row,j+4,fn1.get(key,0))
                sheet.write(row,j+5,precision)
                sheet.write(row,j+6,recall)
                if precision+recall == 0:
                    sheet.write(row,j+7,0)
                else:
                    sheet.write(row,j+7,float(2*precision*recall/(precision+recall)))
                accuracy = float((tp1.get(key,0)+tn1.get(key,0))/float(tp1.get(key,0)+tn1.get(key,0)+fp1.get(key,0)+fn1.get(key,0)))
                # if accuracy == 0:
                #     continue
                sheet.write(row,j+8,accuracy)
                j += 8
            row += 1

        save_path = os.getcwd()+"/Files/Combined/"+t
        workbook_eval.save(save_path+"/Evaluation_Parameters".encode('utf-8')+".xls")

    print("--- %s Minutes ---" % str((time.time() - start_time)/60))


