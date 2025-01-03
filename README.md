# Part-of-Speech Tagging Using HMM and Simple Model

This project implements **Part-of-Speech (POS) tagging** using two approaches:
1. **Simple Model (Maximum Likelihood Estimation)**: Predicts tags for each word independently based on probabilities.  
2. **Hidden Markov Model (HMM - Viterbi Algorithm)**: Predicts the sequence of tags by considering the dependencies between tags.  

The goal is to predict POS tags for words in test sentences and compute the posterior probability of the predicted sequence.

---

## Problem Abstraction  

1. **Task**:
   - Implement Part-of-Speech tagging using:
     - Simple Model (Maximum Likelihood Estimation)
     - Hidden Markov Model (Viterbi Algorithm)

2. **Input**:
   - Training data consisting of sentences with their POS tags.
   - Test sentences requiring POS tag prediction.

3. **Output**:
   - Predicted POS tags for each word in test sentences.
   - Posterior probability of the predicted sequence for both models.

---

## Algorithm Implementation  

### 1. **Simple Model (Maximum Likelihood Estimation)**  
- **Formula**:  
  \[
  P(\text{tag} | \text{word}) = P(\text{word} | \text{tag}) \times P(\text{tag})
  \]
- **Approach**:  
  - For each word:
    - Compute \( P(\text{tag} | \text{word}) \) using precomputed emission probabilities \( P(\text{word} | \text{tag}) \) and tag probabilities \( P(\text{tag}) \).
    - Assign the tag with the maximum probability.
  - Handle unknown words by defaulting to 'noun'.  

---

### 2. **Hidden Markov Model (HMM - Viterbi Algorithm)**  
- **Forward Logic**:
  - Computes the most probable sequence of tags using dynamic programming.
  - Transition probabilities \( P(\text{tag}_i | \text{tag}_{i-1}) \) and emission probabilities \( P(\text{word}_i | \text{tag}_i) \) are calculated for each word and tag pair.
  - Log probabilities are used to prevent underflow.  

- **Backward Logic**:
  - Starts with the most probable tag for the last word.
  - Backtracks through the sequence using pointers to construct the optimal tag path.

- **Unknown Word Handling**:
  - Special 'Unknown' token added to the vocabulary.
  - Emission matrix extended with an extra column to handle unknown words.

- **Posterior Probability**:
  - Stores the joint probability \( P(S, W) \) using the final forward probabilities from the Viterbi matrix.

---

## Implementation Details  

### Training Phase:
- Precomputed the following from the training data:
  1. **Emission Counts**: \( P(\text{word} | \text{tag}) \).  
  2. **Transition Counts**: \( P(\text{tag}_i | \text{tag}_{i-1}) \), including special start `<S>` and end `</S>` tags.  
  3. **Tag Probabilities**: \( P(\text{tag}) \), required for the Simple Model.  

### Program Structure:
1. **Simple Model Implementation**:
   - Compute \( P(\text{tag} | \text{word}) = P(\text{word} | \text{tag}) \times P(\text{tag}) \).
   - Assign the tag with the maximum probability for each word.
   - Handle unknown words by defaulting to 'noun'.

2. **HMM Implementation**:
   - Use dynamic programming to implement the Viterbi Algorithm:
     - **Forward Pass**:
       \[
       P(\text{tag}_i | \text{word}_i, \text{tag}_{i-1}) = P(\text{word}_i | \text{tag}_i) \times P(\text{tag}_i | \text{tag}_{i-1}) \times P(\text{tag}_{i-1})
       \]
       - For each word, compute probabilities for all possible tags and store the maximum probability for each tag.
     - **Backward Pass**:
       - Trace back the path of the most probable tag sequence.
   - Joint probability \( P(S, W) \) is stored as the product of all forward probabilities.

---

## Results  

### Accuracy Scores  
After scoring 2,000 sentences with 29,442 words using the training set (`bc.train`) and test set (`bc.test`), the following metrics were obtained:

| Model            | Words Correct (%) | Sentences Correct (%) |
|-------------------|-------------------|-----------------------|
| **Ground Truth**  | 100.00%            | 100.00%                |
| **Simple Model**  | 93.92%             | 47.55%                 |
| **HMM Model**     | 95.09%             | 54.50%                 |

---

## Key Insights  

1. **Simple Model (Maximum Likelihood Estimation)**:
   - Provides a baseline accuracy of **93.92%** for word-level predictions.
   - Struggles with unknown words and contextual dependencies.

2. **Hidden Markov Model (HMM)**:
   - Achieves higher accuracy (**95.09% word-level**, **54.50% sentence-level**).
   - Utilizes contextual transitions between tags for better predictions.

3. **Posterior Probability**:
   - Both models compute the posterior probability of the predicted tag sequence to evaluate confidence.

---

## Conclusion  

The **Hidden Markov Model (HMM)** outperformed the **Simple Model** in both word-level and sentence-level accuracy by leveraging sequence information and dynamic programming. The results demonstrate the effectiveness of HMM for POS tagging tasks.
