
# Twitter Sentiment Classification

This project implements a deep learning model for classifying the sentiment of tweets using an LSTM-based approach. The notebook processes raw textual data, trains a model, and evaluates its performance.

## Project Workflow

1. **Importing Required Libraries**
   - The project utilizes Python libraries such as TensorFlow, datasets, numpy, pandas, and matplotlib.

2. **Data Preparation**
   - Downloads datasets for training and testing from a specified URL.
   - Splits the data into training, validation, and test sets.
   - Saves processed data in JSONL format for further use.

3. **Model Design**
   - Implements a sequential LSTM model with embedding, LSTM, and dense layers.
   - Optimizes the model using the Adam optimizer.

4. **Training and Evaluation**
   - Trains the model on the prepared dataset.
   - Evaluates the performance using metrics such as accuracy.

5. **Visualization**
   - Includes plots to visualize metrics like loss and accuracy over training epochs.

## Results
- The notebook provides insights into the model's performance on test data and visualizes training progress.

## Requirements
- Python 3.x
- Libraries: TensorFlow, datasets, numpy, pandas, matplotlib

## Usage
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the notebook in Jupyter or any compatible environment.

## File Descriptions
- **TwitterSentimentClassification.ipynb**: Main notebook containing code, explanations, and results.
- **Processed JSONL Files**: Created during data preparation for training and testing.

## Acknowledgments
- Data Source: [GitHub Repository](https://github.com/cblancac/SentimentAnalysisBert)

---
