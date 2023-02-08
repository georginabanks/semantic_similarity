# extract 1
import spacy
nlp = spacy.load('en_core_web_sm')

word1 = nlp('cat')
word2 = nlp('monkey')
word3 = nlp('banana')

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# extract 2
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# extract 3
sentence_to_compare = 'Why is my cat on the car'

sentences = ['where did my dog go',
'Hello, there is my car',
'I\'ve lost my car in my car',
'I\'d like my boat back',
'I will name my dog Diana']

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + ' - ', similarity)

"""
Initially I was quite surprised by the fact that 'cat'
and 'monkey' scored 0.59 similarity. Although both have
fur (I think), both have tails, are mammals, etc.,
I would have thought it would have been slightly lower.
They are not two animals I would have said are similar.

I was also surprised that 'cat' and 'banana' scored 0.22,
as I thought it would be lower. I cannot see similarities
between them other than they may both be found in a house
and their interactions with humans.

I thought 'monkey' and 'banana' would be higher, as they are
two things well associated with each other - monkeys eat
bananas.
"""

"""
When I ran the code using 'en_core_web_sm', I got an error 
for tokens, suggesting I use a larger models instead, which
I wasn't expecting to happen.

The similarity between 'cat' and 'monkey' was 0.65 using the
smaller model, which I wasn't expecting either, however it
does make sense that it would be less 'accurate'.
"""
