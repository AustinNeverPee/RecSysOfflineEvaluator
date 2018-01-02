import re
import numpy as np
import itertools
import pickle
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA

def get_delicious_data():

    #Tag Information
    tag_id_text = defaultdict(lambda: [])
    tag_text_count = defaultdict(lambda: 0) 
    
    for line in open("delicious_data/tags.dat").readlines()[1::]:
    
        tag_id, tag_text = line.split()
        
        tag_text = re.split("_|-|'" , tag_text.strip())
        tag_id_text[tag_id] += tag_text
    
        for single_text in tag_text:
            tag_text_count[single_text] += 1

    kept_tag_texts = {}
    for tag_id in tag_id_text.keys():
        kept_text = []
        for single_text in tag_id_text[tag_id]:
            if tag_text_count[single_text] < 10:
                pass
            else:
                kept_tag_texts[single_text] = tag_text_count[single_text]
                kept_text.append(single_text)
        
        if len(kept_text) == 0:
            tag_id_text.pop(tag_id)
        else:
            tag_id_text[tag_id] = kept_text

    #print sorted(kept_tag_texts.iteritems(), key=lambda (k, v):(v,k))

    #print len(set(list(itertools.chain.from_iterable(tag_id_text.values()))))
    #print "Number of Tags:%d" % (len(tag_id_text.keys()))
    
    ##Bookmark and tags
    #bid_vector_rowid -> find the row id as i-> text_tfidf_matrix[i, :] as the vector for the bookmark
    
    bid_tagid = defaultdict(lambda: [])
    bid_text = defaultdict(lambda: "")

    for line in open("delicious_data/bookmark_tags.dat").readlines()[1::]:
    
        bid, tagid, tagweight = line.split()
    
        bid_tagid[bid].append(tagid)

        try:
            #bid_text[bid] += tag_id_text[tagid]
            if len(tag_id_text[tagid]) == 0:
                #continue
                bid_text[bid] += ""
            else:
                bid_text[bid] += " ".join(tag_id_text[tagid]) + " "
        except KeyError:
            pass
   
    #for bid in bid_text.keys():
    #    print bid_text[bid]

    bid_vector_rowid = {}
    text_lists = []
    rowid = 0
    is_source = []
    for bid in bid_text:
        bid_vector_rowid[bid] = rowid
        text_lists.append(bid_text[bid])
        rowid += 1
        
        is_source.append(False)
        for one_tag_text in bid_text[bid].split():
            if kept_tag_texts[one_tag_text] > 80:
                is_source[-1] = True
                break
    
    #count_vectorizer = CountVectorizer()
    #text_count_matrix = count_vectorizer.fit_transform(text_lists)
    
    tfidf_vectorizer = TfidfVectorizer(decode_error='ignore')
    text_tfidf_matrix = tfidf_vectorizer.fit_transform(text_lists)
    
    ##PCA
    pca = PCA()
    sum_tfidf_matrix = np.sum(text_tfidf_matrix.toarray(), axis=1)
    pca.fit_transform(text_tfidf_matrix.toarray()[sum_tfidf_matrix != 0])
    text_pca_matrix = text_tfidf_matrix.dot(pca.components_[0:25, :].T)
    print "Tag's Tf Idf Matrix Shape", text_pca_matrix.shape
    print "Source Number:%d, Target Number:%d" % (np.sum(is_source), len(is_source) - np.sum(is_source))
    print "Information Explained:%f" % (np.sum(pca.explained_variance_[0:25]) / np.sum(pca.explained_variance_))
    
    #User Id and bookmarks id
    uid_bid = defaultdict(lambda: [])
    for line in open("delicious_data/user_taggedbookmarks.dat").readlines()[1::]:
    
        uid, bid, tid = line.split()[0:3]
   
        try:
            bid_vector_rowid[bid]
            uid_bid[uid].append(bid)
        except KeyError:
            pass


    ##############################################
    for uid in uid_bid:
        #print "Len:%d, Set Len:%d" % (len(uid_bid[uid]), len(set(uid_bid[uid])))
        uid_bid[uid] = list(set(uid_bid[uid]))
    ##############################################
    
    #User Id and uid similarity
    uid_uid = defaultdict(lambda:[])
    for line in open("delicious_data/user_contacts.dat").readlines()[1::]:
    
        uid, cid = line.split()[0:2]
        uid_uid[uid].append(cid)
       
    uid_bid.default_factory = None
    uid_uid.default_factory = None

    return bid_vector_rowid, text_pca_matrix, uid_bid, uid_uid, is_source 

if __name__ == "__main__":

    bid_vector_rowid, text_pca_matrix, uid_bid, uid_uid, is_source = get_delicious_data()
