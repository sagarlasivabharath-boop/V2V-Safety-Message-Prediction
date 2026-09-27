# V2V Transformer Methodology

## 1. Input

The system receives a V2V safety message containing information about a vehicle
situation, such as sudden braking, collision risk, obstacles, traffic, or road
conditions.

## 2. Text Processing

The message text is passed to a Transformer tokenizer.

The tokenizer converts the text into numerical tokens that can be processed by
the Transformer model.

## 3. Transformer Encoding

A pre-trained DistilBERT Transformer is used to obtain a contextual numerical
representation of the V2V message.

The prototype produces a 768-dimensional embedding for each message.

## 4. Criticality Prediction

The Transformer embedding is given to a Logistic Regression classifier.

The classifier predicts one of four criticality levels:

- Low
- Medium
- High
- Critical

## 5. Deadline Prediction

The same Transformer embedding is given to a Ridge Regression model.

The model predicts the expected response deadline in seconds.

## 6. Evaluation

Criticality prediction is evaluated using:

- Accuracy
- Weighted F1-score

Deadline prediction is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## 7. Prototype Dataset

The current prototype uses a small 15-row synthetic V2V safety-message dataset.

The dataset is intended to validate the implementation pipeline and demonstrate
the project concept.

Larger and real-world V2V datasets would be required for robust model
evaluation.