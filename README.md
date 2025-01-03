# Part-of-Speech Tagging Using Simple Model and Hidden Markov Model (HMM)

This project implements **Part-of-Speech (POS) tagging** using two approaches:
1. **Simple Model**: Maximum Likelihood Estimation (MLE)
2. **Hidden Markov Model (HMM)**: Viterbi Algorithm  

The goal is to predict POS tags for words in sentences and calculate the posterior probability of the predicted tag sequence.

---

## Problem Abstraction  

1. **Objective**:  
   - Implement POS tagging using MLE and HMM.  
   - Compare performance and accuracy metrics between the two models.

2. **Input**:  
   - **Training Data**: Sentences annotated with their POS tags (e.g., `bc.train`).  
   - **Test Data**: Sentences requiring POS tag prediction (e.g., `bc.test`).  

3. **Output**:  
   - Predicted POS tags for each word in the test sentences.  
   - Posterior probability of the predicted sequence for both models.  

---

## Algorithm Details  

### 1. **Simple Model (Maximum Likelihood Estimation)**  
- **Formula**:  
  \[
  P(\text{tag} | \text{word}) = P(\text{word} | \text{tag}) \times P(\text{tag})
  \]
- **Implementation**:  
  - For each word, compute probabilities for all possible tags.  
  - Assign the tag with the highest probability.  
  - Handle unknown words by defaulting to `noun`.  

---

### 2. **Hidden Markov Model (Viterbi Algorithm)**  
- **Forward Logic**:  
  - Calculates the most probable tag sequence using dynamic programming.  
  - Transition and emission probabilities are calculated as:  
    \[
    P(\text{tag}_i | \text{word}_i, \text{tag}_{i-1}) = P(\text{word}_i | \text{tag}_i) \times P(\text{tag}_i | \text{tag}_{i-1})
    \]
  - Handles unknown words with a special "Unknown" token.  

- **Backward Logic**:  
  - Backtracks through the sequence to construct the optimal tag path.  
  - Starts with the most probable tag for the last word and reverses the path.  

- **Posterior Probability**:  
  - Computes the joint probability of the sequence \( P(S, W) \) using the forward probabilities from the Viterbi matrix.

---

## Implementation Workflow  

### 1. **Training Phase**  
- Pre-calculate emission probabilities \( P(\text{word} | \text{tag}) \) and transition probabilities \( P(\text{tag}_i | \text{tag}_{i-1}) \).  
- Create a special handling mechanism for unknown words and tags.  
- Extend matrices for initial and end states using `<S>` and `</S>` tags.  

### 2. **Testing Phase**  
- For the **Simple Model**, calculate \( P(\text{tag} | \text{word}) \) for each word in the sentence.  
- For **HMM**, use the Viterbi algorithm to compute the most probable sequence.  

---

## Results  

### Accuracy Scores  
After scoring 2,000 sentences with 29,442 words using `bc.train` and `bc.test`, the following metrics were obtained:

| Model            | Words Correct (%) | Sentences Correct (%) |
|-------------------|-------------------|-----------------------|
| **Ground Truth**  | 100.00%            | 100.00%                |
| **Simple Model**  | 93.92%             | 47.55%                 |
| **HMM Model**     | 95.09%             | 54.50%                 |

---

## Key Takeaways  

1. **Simple Model (MLE)**:  
   - Provides a baseline accuracy of **93.92%** for word-level predictions.  
   - Struggles with unknown words and complex sequences.  

2. **Hidden Markov Model (HMM)**:  
   - Achieves higher accuracy (**95.09% word-level**, **54.50% sentence-level**).  
   - Performs better by considering contextual transitions between tags.  

3. **Posterior Probability**:  
   - Both models calculate probabilities to evaluate sequence confidence.  

---

## Conclusion  

The **Hidden Markov Model (HMM)** outperformed the **Simple Model** in both word-level and sentence-level accuracy. It effectively leverages sequence information, making it more robust for POS tagging tasks.
