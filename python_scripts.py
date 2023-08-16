from transformers import AutoModelForTokenClassification,AutoTokenizer, pipeline
from collections import Counter
import re
import numpy as np

def setup():
    #Get model and tokenizer

    tokenizer = AutoTokenizer.from_pretrained("/workspace/Distilbert_NER")

    # for using the gpu version of this code comment/uncomment this
    # model = AutoModelForTokenClassification.from_pretrained("/workspace/Distilbert_NER").to('cuda:0')
    # pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple",device = 0)

    model = AutoModelForTokenClassification.from_pretrained("/workspace/Distilbert_NER")
    pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    return pipe

def inference(pipe,input):
    
    #Vocab caontains words used for Price Filter
    vocab = [['>','>=', 'more', 'beyond', 'over', 'costlier', 'higher', 'above', 'greater','atleast', 'at least', 'minimum', 'min'],
         ['<','<=','less', 'within', 'under', 'lower ', 'cheaper', 'below', 'lesser','at most', 'atmost', 'maximum', 'max'],
         ['=', 'equivalent', 'parallel', 'equal', 'similar', 'akin', 'comparable', 'for']]

    f = open('/workspace/big.txt','w+')
    for i in vocab:
        for j in i:
            f.write(j)
            f.write(' ')
    f.write('than to')

    # All functions for Spell Check
    def words(text): \
        return re.findall(r'\w+', text.lower())

    def read():
        f.seek(0)
        return f.read()

    WORDS = Counter(words(read()))
    WORDLIST = list(WORDS.keys())

    def P(word, N=sum(WORDS.values())):
        "Probability of `word`."
        return WORDS[word] / N

    def correction(word):
        "Most probable spelling correction for word."
        return max(candidates(word), key=P)

    def candidates(word):
        "Generate possible spelling corrections for word."
        return (known([word]) or known(edits1(word)) or known(edits2(word)) or ['NA'])

    def known(words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in WORDS)

    def edits1(word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in edits1(word) for e2 in edits1(e1))

    #Call the ML Pipeline
    result = pipe(input)

    #Get entities spearately
    comp = ''
    pr = ''

    for j in range(len(result)):
        if(result[j]['entity_group']=='comparison'):
            comp += result[j]['word'] + ' '
        if(result[j]['entity_group']=='price'):
            pr += result[j]['word'] + ' '

    #Perform Spell Check and Classification
    comp = comp.replace('than','').replace('to','').strip()
    comp = ' '.join([correction(i) for i in comp.split()])
    term = comp
    comp = comp.split()
    occurrence_lists = np.array([[[k in i for i in vocab[i]].count(True) for i in range(3)] for k in comp])
    pos = -1 if (np.sum(occurrence_lists,axis=0)==[0,0,0]).all() else np.argmax(np.sum(occurrence_lists,axis=0))
    comp = '>' if pos==0 else '<' if pos==1 else '='

    result = [term, pr, comp]

    return result
