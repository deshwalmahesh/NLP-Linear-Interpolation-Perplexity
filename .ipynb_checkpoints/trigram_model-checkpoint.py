import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2019
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(inp, n):
    """
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    if type(inp) == str:
        sequence = inp.split()
    else:
        sequence = inp.copy()
        
    end = len(sequence)
    error = 0
    result = []
    start = 0

    
    sequence.insert(0,'START')
    sequence.append('STOP')
    end+=2
    
    if n==1:
        return sequence
    
    else:
        while start+n<end+1:
            result.append(tuple(sequence[start:start+n]))
            start+=1
        return result


class TrigramModel(object):
    
    def __init__(self, corpusfile):
        
        self.total_words = 0
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        
        one_g = []
        two_g = []
        three_g = []
        for sequence in corpus:
            self.total_words += len(sequence)
            one_g.extend(get_ngrams(sequence,1))
            two_g.extend(get_ngrams(sequence,2))
            three_g.extend(get_ngrams(sequence,3))
            
        self.unigramcounts = Counter(one_g)
        self.bigramcounts = Counter(two_g)
        self.trigramcounts = Counter(three_g)

        return None
    
    
    def raw_trigram_probability(self,trigram):
        """
        Returns the raw (unsmoothed) trigram probability
        """
        assert len(trigram)==3, "Input should be 3 words"
        if self.bigramcounts[trigram[:2]] == 0:
            return 0
        else:
            return self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]

    
    def raw_bigram_probability(self, bigram):
        """
        Returns the raw (unsmoothed) bigram probability
        """
        assert len(bigram)==2, "Input should be 2 words"
        if self.unigramcounts[bigram[0]] == 0:
            return 0
        else:
            return self.bigramcounts[bigram]/self.unigramcounts[bigram[0]]
        
    
    def raw_unigram_probability(self, unigram):
        """
        Returns the raw (unsmoothed) unigram probability.
        """
        uni = []
        uni.append(unigram)
        assert len(uni)==1, "Input should be only 1 word"
        return self.unigramcounts[unigram]/self.total_words


    def smoothed_trigram_probability(self, trigram):
        """
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        assert len(trigram)==3, "Input should be 3 words"
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        u,v,w = trigram[0],trigram[1],trigram[2]
        prob =  (lambda1*self.raw_unigram_probability(w))+\
        (lambda2*self.raw_bigram_probability((v,w)))+\
        (lambda3*self.raw_trigram_probability((u,v,w)))
        return prob
    
    
    def sentence_logprob(self, sentence):
        """
        Returns the log probability of an entire sequence.
        """
        from math import log2
        if type(sentence) == str:
            sentence = sentence.split()
        tri_g = get_ngrams(sentence,3)
        sent_prob = 0.0
        for tri_tuple in tri_g:
            sent_prob += log2(self.smoothed_trigram_probability(tri_tuple))
            
        return sent_prob
    
    
    def perplexity(self, corpus):
        """ 
        Returns the log probability of an entire sequence.
        """
        if type(corpus) == 'str':
            corpus = corpus_reader(corpus, self.lexicon)
        
        total_log_prob = 0.0
        for sentence in corpus:
            total_log_prob += self.sentence_logprob(sentence)
        
        l = total_log_prob/self.total_words
        
        return float(2**(-l))
      


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            
            if pp < pp_low:
                correct += 1
            else:
                correct -= 1
            total += 1
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp_high = pp = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        
            if pp < pp_high:
                correct += 1
            else:
                correct -= 1
            
            total += 1
        
        return float(correct/total)
    

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

