# Part-of-Speech Tagging Using HMM and Simple Model

This project implements Part-of-Speech (POS) tagging using two models:  
1. **Simple Model** (Maximum Likelihood Estimation).  
2. **Hidden Markov Model** (HMM) with the Viterbi Algorithm.

---

## Problem Abstraction

### Task:
- Predict the POS tags for words in test sentences using:
  - A **Simple Model** based on Maximum Likelihood Estimation.
  - A **Hidden Markov Model** using the Viterbi Algorithm.

### Inputs:
- **Training Data**: Sentences annotated with their corresponding POS tags.
- **Test Data**: Sentences requiring POS tag predictions.

### Outputs:
- **Predicted POS Tags**: One tag for each word in the test sentences.
- **Posterior Probability**: Probability of the predicted sequence of tags.

---

## Algorithm Details

### 1. Training Phase:
For both models, probabilities and counts are precomputed from the training data (`bc.train`):
- **Emission Counts/Matrix**: \( P(\text{word}|\text{tag}) \).
- **Transition Counts/Matrix**: \( P(\text{tag}_i|\text{tag}_{i-1}) \), including start (`<S>`) and end (`</S>`) tags.
- **Tag Probabilities**: \( P(\text{tag}) \), required for the Simple Model.

---

### 2. Simple Model:
Implements Maximum Likelihood Estimation to compute \( P(\text{tag}|\text{word}) \):
- **Logic**:
  - \( P(\text{tag}|\text{word}) = P(\text{word}|\text{tag}) \times P(\text{tag}) \).
  - For each word, assign the tag with the highest probability.
  - Handles unknown words by defaulting to the 'noun' tag.
- **Posterior Probability**:
  - Calculated as the sum of log probabilities for the predicted sequence.

---

### 3. Hidden Markov Model (HMM):
Uses the **Viterbi Algorithm** for sequence prediction:
- **Transition Probabilities**: Between tags \( P(\text{tag}_i|\text{tag}_{i-1}) \).
- **Emission Probabilities**: Between tags and words \( P(\text{word}_i|\text{tag}_i) \).
- **Logarithmic Probabilities**:
  - Avoids underflow by working in log space for both transition and emission probabilities.

#### Implementation Steps:
1. **Initialization**:
   - Start with initial probabilities \( P(\text{tag}_0|\text{<S>}) \times P(\text{word}_0|\text{tag}_0) \).
   - Handle unknown words using a special 'Unk' token.

2. **Forward Pass**:
   - For each word in the sentence:
     - Calculate \( P(\text{tag}_i|\text{word}_i, \text{tag}_{i-1}) = P(\text{word}_i|\text{tag}_i) \times P(\text{tag}_i|\text{tag}_{i-1}) \times P(\text{tag}_{i-1}) \).
     - Store the maximum probability for each current tag.

3. **Backward Pass**:
   - Trace back using pointers to reconstruct the optimal sequence.
   - Reverse the sequence to obtain the correct order.

4. **Joint Probability**:
   - Compute the joint probability \( P(S, W) \), stored as \( T[\text{maxLastTagIndex}, \text{lastWordIndex}] \).

#### Posterior Probability:
- Retrieved directly from the joint probability computed by Viterbi.

---

## Results

### Evaluation Metrics:
The models were evaluated using 2,000 sentences (29,442 words) from `bc.test` and scored against ground truth tags.

| **Model**         | **Words Correct (%)** | **Sentences Correct (%)** |
|--------------------|-----------------------|---------------------------|
| **Ground Truth**   | 100.00%              | 100.00%                   |
| **Simple Model**   | 93.92%               | 47.55%                    |
| **HMM Model**      | 95.09%               | 54.50%                    |

### Observations:
- The **HMM Model** outperformed the Simple Model in both word-level and sentence-level accuracy.
- The HMM's use of sequence-level information improves predictions, especially for longer sentences.

### Key Insights:
- **HMM Accuracy**:
  - **Word-Level**: 95.09%.
  - **Sentence-Level**: 54.50%.
- The Simple Model is less effective due to its independent treatment of words, whereas the HMM incorporates context through transitions.

---

## Conclusion
The Hidden Markov Model (HMM) with the Viterbi Algorithm provides better accuracy for POS tagging compared to the Simple Model. Its ability to capture dependencies between consecutive tags makes it more suitable for sequence-based tasks.
