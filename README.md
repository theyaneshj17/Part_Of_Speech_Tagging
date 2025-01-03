# Part-of-Speech Tagging using HMM and Simple Model
Part-of-Speech Tagging using Simple Model (Maximum Likelihood Estimation) and Hidden Markov Model (Viterbi Algorithm)

### Abstraction of the Problem

1. **Task**: Implement Part-of-Speech tagging using:
   - Simple Model (Maximum Likelihood Estimation)
   - Hidden Markov Model (Viterbi Algorithm)

2. **Input**: 
   - Training data consisting of sentences with their POS tags
   - Test sentences requiring POS tag prediction

3. **Output**:
   - Predicted POS tags for each word in test sentences
   - Posterior probability of the predicted sequence

### Algorithm Implementation

1. **Simple Model**:
   - Maximum Likelihood Estimation
   - P(tag|word) = P(word|tag) * P(tag)
   - Assign tag based on the maxium probabilities for each and word, tag

2. **Hidden Markov Model**:
   - Implemented Viterbi Algorithm
   - Created transition probabilities between previous tag and current tag 
   - Created emission probabilities between tags and word


### Implementation Details

1. **Training Phase**:
   - During this, precalculate all the required probabilites / count for the data available in bc.train for both algorithms

   - Calculate emission counts and emission matrix (word|tag)
   - Calculate transition counts and transition matrix(tag₂|tag₁) (Handles, start \<S> and end \</S> tags)
   - Calculate P(Si) and P(Wi|Si), required for Simple Bayes.


2. **Simple Model Implementation**:
   - For each word:
     - Computes P(Si|Wi) using P(Wi|Si) * P(Si), calculated above
     - Selects tag with maximum probability
     - Handles unknown words by defaulting to 'noun'

3. **HMM Implementation**:
  - Implements dynamic programming
  - Constructs log probabilities to prevent underflow for both transition and emission probabilities:
  - Handle Unknown Words:
    - Add special 'Unk' token to vocabulary
    - Extend emission matrix with extra column
  - Handle initial probabilities for all tag:
    - P(tag|word) = P(word|tag) * P(tag|\<S>)
    - EmissionProb = EmissionMatrixModified[j][WordIndex]
    - TransitionProb = TransitionMatrixModified[StartIndex][tagIndex[tag]]
    - T[j, 0] = EmissionProb + TransitionProb

  - Forward Logic:
    - P(tag_i|word_i, tag_{i-1}) = P(word_i|tag_i) * P(tag_i|tag_{i-1}) * P(tag_{i-1})
    - EmissionProb = EmissionMatrixModified[currTagIndex][WordIndex]
    - TransitionProb = TransitionMatrixModified[prevTagFullIdx][currTagFullndex]
    - priorProb = T[prevTagIndex, i-1]
    - currprob = priorProb + TransitionProb + EmissionProb
    - Store maximum probability for each current tag
  - Backward Logic:
    - Start with most probable tag of last word:
    - Follow backpointers to reconstruct optimal 
    - Reverse path to get correct order:

  - Joint Probability:
    - Store final joint probability P(S,W):
    - self.JointProb = T[maxLastTagIndex, lastWordIndex]


4. **Posterior Probability**:
   - Simple Model: Computes sum of log probabilities
   - HMM: Returns joint probability from Viterbi

# Results

### Accuracy Scores
After scoring 2,000 sentences with 29,442 words using bc.train and bc.test, the following accuracy metrics were obtained:

| Model            | Words Correct (%) | Sentences Correct (%) |
|-------------------|-------------------|-----------------------|
| **Ground Truth**  | 100.00%            | 100.00%                |
| **Simple Model**  | 93.92%             | 47.55%                 |
| **HMM Model**     | 95.09%             | 54.50%                 |

### Conclusion
The Hidden Markov Model (HMM) outperformed the Simple Model in both word-level and sentence-level accuracy. 

- **HMM Accuracy**:  
  - **Word-level**: 95.09%  
  - **Sentence-level**: 54.50%  

