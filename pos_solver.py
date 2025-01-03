
import random
import math
from collections import defaultdict
import numpy as np



class Solver:
    
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    
    def posterior(self, model, sentence, label):

        if model == "Simple":

            prob = 0

            for i, word in enumerate(sentence):

                word = word

                tag = label[i]
                
                if word in self.wordTagProb and tag in self.wordTagProb[word]:
                    # P(Wi|Si) * P(Si))
                    prob = prob + math.log(self.wordTagProb[word][tag] * self.tagProb[tag])
                else:
                    
                    prob += math.log(1e-10) #Unseen words
            
            return prob
        
        elif model == "HMM":    
                        
                return self.JointProb #Already computed in hmm_viterbi
        else:
            
            print("Unknown algo!")


    def createTransitionMatrix(self):
        
        allTags = sorted(self.tagCounts.keys()) #Include all tags even <S>, </S>
        
        numTags = len(allTags)

        TransitionMatrix = np.zeros((numTags,numTags))
        
        for i in range(numTags):

            for j in range(numTags):

                count = 0

                key = (allTags[i], allTags[j])

                if key in self.transitionCounts:

                    count = self.transitionCounts[key]

                countPrevTag = self.tagCounts[allTags[i]]


                if countPrevTag > 0:

                    TransitionMatrix[i,j] = (count )/(countPrevTag )
                
                else:

                    TransitionMatrix[i,j] = 0


        return TransitionMatrix

    def createEmissonMatrix(self):

        allTags = [tag for tag in sorted(self.tagCounts.keys()) if tag not in ['<S>', '</S>']] #Ignore, start and end Tags

        numTags = len(allTags)

        vocab_list = list(self.vocab.keys())

        num_words = len(vocab_list)
        
        EmissionMatrix = np.zeros((numTags, num_words))

        for i in range(numTags):

            for j in range(num_words):

                count = 0

                key = (allTags[i],  vocab_list[j])

                if key in self.emissionCounts:

                    count = self.emissionCounts[key]

                countTag = self.tagCounts[allTags[i]]

                EmissionMatrix[i,j] = (count )/(countTag )      

        return EmissionMatrix


    def train(self, data):


        self.emissionCounts = defaultdict(int)

        self.transitionCounts = defaultdict(int)

        self.tagCounts = defaultdict(int) #list of tags--{tag, count}

        self.wordCounts = defaultdict(int) #list of words--{word, count}

        self.vocab = defaultdict(int) 

        self.tagCounts['<S>'] = len(data) #Start Tag

        self.tagCounts['</S>'] = len(data) #End Tag
        
        index = 0
        
        self.totalTags = 0
        
        self.wordTagCounts ={}

        for sentence in data:
            
            words = sentence[0]
            
            tags = sentence[1]
            
            prevTag = '<S>'

            for word, tag in zip(words, tags):

                self.transitionCounts[(prevTag, tag)] += 1
                
                self.tagCounts[tag] += 1
                
                prevTag = tag
                
                self.wordCounts[word] += 1
                
                self.totalTags += 1

                if word not in self.wordTagCounts:
                    
                    self.wordTagCounts[word] = {}

                self.wordTagCounts[word][tag] = self.wordTagCounts[word].get(tag, 0) + 1

                
                if word not in self.vocab:

                    self.vocab[word] = index

                    index += 1
                    
                if tag != '<S>':  

                    self.emissionCounts[(tag, word)] += 1

            self.transitionCounts[(prevTag, '</S>')] += 1
        
        self.tagProb = {}

        self.wordTagProb ={}
        
        #P(Si)
        for tag in self.tagCounts:
            
            self.tagProb[tag] = self.tagCounts[tag] / self.totalTags

        #P(Wi|Si)   
        for word in self.wordTagCounts:

            self.wordTagProb[word] = {}

            for tag in self.wordTagCounts[word]:

                self.wordTagProb[word][tag] = self.wordTagCounts[word][tag] / self.tagCounts[tag]

        
        self.states = sorted(self.tagCounts.keys())

        self.numTags = len(self.states)

        self.TransitionMatrix = self.createTransitionMatrix() #Creating Transition Matrix

        self.EmissionMatrix = self.createEmissonMatrix() #Creating Emission Matrix



        #print('TransitionMatrix')
        #print(self.TransitionMatrix)
        #import pandas as pd
        # Convert to pandas DataFrame
        #allTags = [tag for tag in sorted(self.tagCounts.keys()) if tag not in ['<S>', '</S>']]
        #EmissionMatrix_df = pd.DataFrame(self.EmissionMatrix, index=allTags, columns=list(self.vocab.keys()))
        # Display the DataFrame
        #print(EmissionMatrix_df)


    def simplified(self, sentence):
        
        result = []
        
        for word in sentence:

            max_prob =float('-inf') 

            modifiedTag = "noun" #For unknown words / not in training, will assign it with noun
            
          
            if word in self.wordTagProb:

                for tag in self.wordTagProb[word]:

                    # P(Si|Wi) 
                    prob = (math.log(self.wordTagProb[word][tag]) + 
                            
                            math.log(self.tagProb[tag]))
                    
                    if prob > max_prob:

                        max_prob = prob

                        modifiedTag = tag
            
            result.append(modifiedTag)
        
        return result

    def hmm_viterbi(self, sentence):

        TransitionMatrixModified = np.log(self.TransitionMatrix + np.finfo(float).eps)  # TransitionMatrixdd small epsilon to avoid log(0)
       
        EmissionMatrixModified = np.log(self.EmissionMatrix + np.finfo(float).eps)
        
        #print('EmissionMatrixModified')
        #print(EmissionMatrixModified)
        
        #print('self.states', self.states)


        wordIndex = {}  

        for index, word in enumerate(self.vocab):

            wordIndex[word] = index  


        #print(wordIndex)


        tagIndex = {}  

        for index, tag in enumerate(self.states):

            tagIndex[tag] = index  


        #print(tagIndex)


        V = len(self.states) - 2  # Exclude <S> and </S>

        numWords = len(sentence)

        T = np.full((V, numWords), -np.inf) 

        backtrack = np.zeros((V, numWords), dtype=int)

        alltags = [tag for tag in self.states if tag not in ['<S>', '</S>']]


        alltagsIndex = {}  

        for index, tag in enumerate(alltags):

            alltagsIndex[tag] = index  


        #Handling wordIndex not present scenario

        UnknownIndex = len(self.vocab) 

        wordIndex['Unk'] = UnknownIndex

       
        EmissionMatrixModified = np.hstack((EmissionMatrixModified, np.zeros((len(alltags), 1))))

      
        StartIndex = tagIndex['<S>']

        word = sentence[0]

        WordIndex = wordIndex.get(word, UnknownIndex)
	
        
        #Initial probabilites
        for tag in alltags:

            j = alltagsIndex[tag]

            # P(tag|word) = P(word|tag) * P(tag|<S>) 

            EmissionProb = EmissionMatrixModified[j][WordIndex]

            #print(EmissionProb,'EmissionProb', 'j',j,WordIndex,EmissionMatrixModified[j][4])
            
            TransitionProb = TransitionMatrixModified[StartIndex][tagIndex[tag]]
            
            #print(P({tag}|{word}, <S>)
            T[j, 0] = EmissionProb + TransitionProb
              
        for i in range(1, numWords):

            word = sentence[i]

            WordIndex = wordIndex.get(word, UnknownIndex)
            
            for currTag in alltags:

                currTagIndex = alltagsIndex[currTag]

                maxlogProb = -np.inf

                maxPrevTag = -1
                
                for prevTag in alltags:

                    prevTagIndex = alltagsIndex[prevTag]

                    prevTagFullIdx = tagIndex[prevTag]

                    currTagFullndex = tagIndex[currTag]
                    
                    
                    EmissionProb = EmissionMatrixModified[currTagIndex][WordIndex]
                    
                    TransitionProb = TransitionMatrixModified[prevTagFullIdx][currTagFullndex]
                    
                    priorProb = T[prevTagIndex, i-1]
                    

                    currprob = priorProb + TransitionProb + EmissionProb
                    #print(f"log P({currTag}|{word}, {prevTag}) = {EmissionProb:.4f} + {TransitionProb:.4f} + {priorProb:.4f} = {currprob:.4f}")
                    
                    if currprob > maxlogProb:

                        maxlogProb = currprob

                        maxPrevTag = prevTagIndex
                
                T[currTagIndex, i] = maxlogProb

                backtrack[currTagIndex, i] = maxPrevTag


        self.maxPath = []
        
        lastWordIndex = numWords - 1

        maxLastTagIndex = np.argmax(T[:, lastWordIndex])

        self.maxPath.append(alltags[maxLastTagIndex])
        

        currentTagIndex = maxLastTagIndex

        for i in range(lastWordIndex, 0, -1):

            currentTagIndex = backtrack[currentTagIndex, i]

            self.maxPath.append(alltags[currentTagIndex])

        
    
        self.maxPath.reverse()
        
        # joint probability P(S,W)
        self.JointProb = T[maxLastTagIndex, lastWordIndex]

        return  self.maxPath
     

    def solve(self, model, sentence):
        
        if model == "Simple":
            
            return self.simplified(sentence)
        
        elif model == "HMM":
            
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
