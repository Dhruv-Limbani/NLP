# -*- coding: utf-8 -*-
"""
Created on Thu May 12 19:17:06 2022

@author: Dell
"""

import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re

paragraph = """I have three visions for India. In 3000 years of our history, people from all over the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards, The Greeks, the Turks, the Moguls, the Portuguese, the British, the French, the Dutch, all of them came and looted us, took over what was ours. Yet we have not done this to any other nation. We have not conquered anyone. We have not grabbed their land, their culture, their history and Tried to enforce our way of life on them. Why? Because we respect the freedom of others.
That is why my first vision is that of FREEDOM. I believe that India got its first vision of this in 1857, when we started the war of Independence. It is this freedom that we must protect and nurture and build on. If we are not free, no one will respect us.
My second vision for India's DEVELOPMENT, For fifty years we have been A developing nation. It is time we see ourselves as a developed nation. We are among top 5 nations of the world in terms of GDP. We have 10 percent growth rate in most areas. Our poverty levels are falling. Our achievements are being globally recognized today. Yet we lack the self-confidence to see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
I have a THIRD vision. India must stand up to the world. Because I believe that, unless India stands up to the world, no one will respect us. Only strength respects strength. We must be strong not only as a military power but also as an economic power. Both must go hand-in-hand. My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of space, Professor Satish Dhawan, who succeeded him and Dr.Brahm Prakash, father of nuclear material. I was lucky to have worked with all three of them closely and consider this the great opportunity of my life.
I see four milestones in my career:"""

text = re.sub(r'\[[0-9]*\]',' ', paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ', text)
text = re.sub(r'\s+',' ',text)

sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

model = Word2Vec(sentences, min_count=1) # if word occurs less than 1 then skip that  

# words = model.wv.vocab # old method. will show an error . comment out before executing entire file

# words = list(model.wv.index_to_key)

words = model.wv.key_to_index

vector = model.wv['war'] # floating word vector

similar = model.wv.most_similar('freedom')
