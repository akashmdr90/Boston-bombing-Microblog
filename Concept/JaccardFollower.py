# coding: utf-8
__author__ = 'sherlock'
import xlrd
import xlwt
import numpy as np
import SETTINGS
import time
import os
import operator
import util as ut


# if __name__ == "__main__":
def main():
    print "Started---------------------------"
    start_time = time.time()
    thresh = ["10","15", "20", "25"]
    followerDict = {}
    workbook = xlrd.open_workbook(SETTINGS.dataset_folder+"dataset.xlsx")
    stop_words = open(SETTINGS.stop_words_file).read().splitlines()
    tweets, num_rows, tweetLabel, followerDict = ut.tweet_from_source_with_Followers(workbook, stop_words)
    rumor_category = {"NA","R1","R3","R4","R5","R7","All"}
    conceptDict = ut.readConceptDictionary()
    for th in thresh:
        rumor_type = {'Type','Description'}

        for t in sorted(rumor_type):
            scores = {}
            rumor_list = ut.get_rumor_list(workbook,t, stop_words)
            rumor_word_count = {i : {} for i in range(rumor_list.__len__())}
            k = 0
            wordnetDict = {}
            lookedInWordnet = set()
            for rumor in rumor_list:
                word_array = rumor.split()
                word_dict = {word : 0 for word in rumor.split()}
                rumor_word_count[k] = word_dict
                k += 1

            threshold = np.arange(0,0.5,0.01)
            original_rumor_word_count = rumor_word_count
            word_count_storage = {}
            for val in threshold:
                i = 0
                print "Threshold Now: "+str(val)
                # workbook_for_jaccard = xlwt.Workbook()
                # sheet = workbook_for_jaccard.add_sheet("Jaccard Labels")
                rumor_word_count = original_rumor_word_count
                labeled_tweets, rumor_word_count = ut.labelDataJaccardConcept(tweets, rumor_word_count, val, rumor_list, conceptDict, th, followerDict)
                word_count_storage[val] = rumor_word_count

                workbook_new = xlwt.Workbook()
                sheet = workbook_new.add_sheet("confusion Matrix")
                sheet.write(0,0,"Tweets")
                sheet.write(0,1,"Category")
                sheet.write(0,2,"True Value")

                tp = {}
                tn = {}
                fp = {}
                fn = {}
                excel = {}
                for key in labeled_tweets:
                    cat = None
                    if key != 0 and key != 2 and key != 3 and key != 4 and key != 6:
                        cat = "NA"
                    elif key == 0:
                        cat = "R1"
                    elif key == 2:
                        cat = "R3"
                    elif key == 3:
                        cat = "R4"
                    elif key == 4:
                        cat = "R5"
                    elif key == 6:
                        cat == "R7"

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
                # print tweets.__len__()
                # for tweet in tweets:
                #     cc = excel.get(tweet)
                #     sheet.write(row,0,tweet.decode('ISO-8859-1'))#.decode('utf-8')
                #     sheet.write(row,1,cc)
                #     sheet.write(row,2,tweetLabel.get(tweet))
                #     row += 1
                # save_path = os.getcwd()+"/Files/Jaccard/"+t
                # workbook_new.save(save_path+"/out_"+str(val).encode('utf-8')+".xls")
                i += 1

            sorted_scores = sorted(scores.items(), key = operator.itemgetter(0))
            # cPickle.dump(word_count_storage, open("word_count_storage_Graph_"+str(t)+".p", "wb"))
            # cPickle.dump(scores, open("scores_Graph_"+t+".p", "wb"))
            # scores = cPickle.load(open("scores_Jacc"+t+".p", "rb"))
            ut.calculations(sorted_scores,rumor_category,t, th, "Jaccard")

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
                    # sheet.write(row,j+8,float((tp1.get(key,0)+tn1.get(key,0))/(tp1.get(key,0)+tn1.get(key,0)+fp1.get(key,0)+fn1.get(key,0))))
                    accuracy = float((tp1.get(key,0)+tn1.get(key,0))/float(tp1.get(key,0)+tn1.get(key,0)+fp1.get(key,0)+fn1.get(key,0)))
                    sheet.write(row,j+8,accuracy)
                    j += 8
                row += 1

            # save_path = os.getcwd()+"/Files/Jaccard/"+t
            save_path = os.getcwd()+"/Files/Jaccard/"+th+"/"+t
            workbook_eval.save(save_path+"/Evaluation_Parameters".encode('utf-8')+".xls")

    print("--- %s Minutes ---" % str((time.time() - start_time)/60))
