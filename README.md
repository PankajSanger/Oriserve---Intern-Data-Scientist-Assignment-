# Oriserve - Intern Data Scientist Assignment Report


## Task: Subtheme Sentiment Analysis
### Problem Statement Overview
The task involves developing an approach to identify subthemes along with their respective sentiments from customer reviews. Subtheme sentiments represent sentiments towards specific aspects or problems within the reviews, and the goal is to accurately identify these subthemes and their associated sentiments.

## Approach
For this task, I used a deep learning-based approach using Long Short-Term Memory (LSTM) networks, which are well-suited for sequential data such as text. Here's an overview of the approach:

### Model Architecture
- LSTM Layers: Stacked LSTM layers to learn sequential patterns in the text data.
- Dropout Layers: Regularization technique to prevent overfitting by randomly dropping a fraction of input units.
- Dense Layers: Fully connected layers for classification.
  
### Motivation
- Sequence Modeling: LSTMs are well-suited for capturing sequential dependencies in text data, making them effective for sentiment analysis tasks.

## Code Explanation
1. **MulticlassBinaryLSTM Class:** This class encapsulates methods for loading datasets, preprocessing data, training LSTM models, generating predictions, and evaluating model performance.
2. **Initialization:** Upon instantiation, the MulticlassBinaryLSTM object is initialized with the path to the dataset CSV file, laying the groundwork for subsequent operations.
3. **Dataset Loading and Preprocessing:** The load_dataset() method fetches the dataset from the specified CSV file, while preprocess_data() performs essential preprocessing steps such as sentiment filtering, column processing, and handling missing values.
4. **Text Preprocessing:** The preprocess_text() function preprocesses text data by converting it to lowercase, removing punctuation, correcting spelling, removing stopwords, and lemmatizing words using Spacy, ensuring optimal data quality for analysis.
5. **Tokenization:** Utilizing Keras Tokenizer, the tokenize_sequences() method tokenizes sequences within the dataset, converting textual data into numerical sequences, a crucial step in preparing data for LSTM model training.
6. **Model Creation and Training:** The create_and_train_lstm_model() function crafts and trains LSTM models using the Sequential API from TensorFlow/Keras, incorporating embedding and LSTM layers. The train_models() method further trains LSTM models for each target column, split data, ensuring comprehensive model coverage.
7. **Prediction Generation and Binarization:** Predictions are generated using the trained LSTM models via the generate_predictions() method and subsequently binarized based on a specified threshold, as implemented in the binarize_predictions() function.
8. **Evaluation:** The calculate_accuracy() function computes accuracy between true and predicted labels, while the evaluate_models() method assesses model performance, including accuracy calculation and training history visualization.
9. **Plotting:** To provide visual insights into model training, the plot_history() function plots training history, showcasing loss and accuracy trends for each target column.
10. **Main Function:** The main function serves as the entry point, orchestrating the entire sentiment analysis process—from data loading to model evaluation—culminating in a robust sentiment analysis framework.

## Future Improvements
- Experiment with different LSTM architectures (e.g., bidirectional LSTM, stacked LSTM) to improve model performance.
- Fine-tune hyperparameters such as the number of LSTM units, dropout rate, and sequence length.
- Undersampling: Undersampling techniques can be applied to balance the classes, which may improve the model's performance.
- Incorporating Attention Mechanisms: Attention mechanisms can be integrated into the LSTM model to give more weight to relevant words or tokens in the input sequence, potentially improving the model's interpretability and performance.

## Result
The LSTM-based model achieved promising results on the test dataset, accurately identifying subthemes and their sentiments in customer reviews. The accuracy of the model indicates its effectiveness in classifying the subthemes and sentiments.
### Predictions
| Subtheme                | Accuracy  |
|-------------------------|-----------|
| Advisor Agent Service   | 0.6847    |
| Balancing               | 0.75      |
| Booking Confusion       | 1.0       |
| Call Wait Time          | 1.0       |
| ...                     | ...       |
| Value for Money         | 0.9683    |
| Wait Time               | 0.6790    |

## Conclusion
The code successfully implements the LSTM-based approach for subtheme sentiment analysis, achieving an average accuracy of approximately 0.867. This indicates that the model is effective in classifying subthemes and sentiments in customer reviews. Further experimentation and fine-tuning may improve the model's performance and generalization capabilities.

---

**Name:** Pankaj Sanger  
**Email ID:** pankajsanger32@gmail.com  
**Mob:** 8527613026
