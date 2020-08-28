from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = 'meet joe black  reviewed on nov  2798  starring brad pitt  anthony hopkins  claire forlani in meet joe black  brad pitt plays death  thats all that really needs to be said  but nevertheless  i will provide the three of you that have seemingly been living in a cave with a plot description  death decides to take a holiday  what with all the rigors of soulcollecting and all  and forces anthony hopkins into showing him what its like to be human  death assumes the body of brad pitt  and from there  much trouble ensues  for one thing  deathaspitt falls in love with hopkins daughter  played by claire forlani  obviously this enrages hopkins  because really  what can death offer  besides eternal damnation  of course  there is also a subplot about forlanis exboyfriend  she dumps him for pitt  trying to take over hopkins company   meet joe black runs just under three hours  ive always thought that such obscene running times should be limited to historical epics   meet joe black is neither historical nor is it an epic  though i get the feeling martin brest  the director  desperately wants it to be  every single scene in the movie goes on about 34 minutes too long  and the ending takes about 20 minutes longer than it really should  a severe editing job could have made this movie excellent  instead of just good  which is what it is  pitt  an actor i normally loathe  is actually quite engaging as death  i believed his performance  and i could see why forlanis character would fall in love with him  i have to agree with roger ebert  though  who found it hard to believe that an entity thats been around for all time wouldnt know what peanut butter was  that has nothing to do with pitts performance  of course  but it is a little distracting  hopkins gives his usual excellent performance  hes able to portray the angst of a man who knows he has very little time left  without making him an obnoxious whiner  and in her first major studio role  claire forlani is surprisingly good  she has a sweet tenderness that allows the audience to instantly root for her  so   meet joe black  is a good movie hampered by its ridiculous running time  had the film been cut by an hour or so  i have no doubt that i would be calling it one of the best movies of the year in this review'
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)