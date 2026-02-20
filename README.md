# Toxic Comment Classification - Anish Abhyankar

## Introduction About the Data :

**The dataset**  
The goal is to **classify toxicity level** of social media comments (Multi-class Classification).  
There are **12 features**:

**Input Features:**
- `comment`: Text content of the comment
- `upvote`: Number of upvotes received
- `downvote`: Number of downvotes received  
- `emoticon_1`: Positive emoticons count
- `emoticon_2`: Sad emoticons count
- `emoticon_3`: Angry emoticons count
- `if_1`: Flag 1 (sensitive content indicator)
- `if_2`: Flag 2 (violation indicator)
- `race`: Race-related flag
- `religion`: Religion-related flag
- `gender`: Gender-related flag
- `disability`: Disability-related flag

**Dataset Source:** https://www.kaggle.com/competitions/comment-category-prediction-challenge/data

**Class Distribution:** Highly imbalanced (50% Clean, 30% Severe Toxic, 14% Toxic, 5% Threat)

## Localhost Deployment Link :

**Flask Web App:** http://127.0.0.1:5000/

Class Distribution:** Highly imbalanced (50% Clean, 30% Severe Toxic, 14% Toxic, 5% Threat)

## Localhost Deployment Link :

**Flask Web App:** http://127.0.0.1:5000/




## API Testing (Local)

**Prediction Endpoint:** http://127.0.0.1:5000/predictdata

## Approach for the project

### Data Ingestion :
- Raw CSV data loaded using pandas
- Train/test split (80/20) saved as artifacts/train.csv, artifacts/test.csv
- Logging and exception handling implemented

### Data Transformation :
**ColumnTransformer Pipeline created:**


### Model Training :
- **Base model tested:** LogisticRegression(multi_class='multinomial')
- **Class weights:** `balanced` for imbalanced dataset
- **Best model:** LogisticRegression(C=1.0, class_weight='balanced')
- **Metrics:** 79% Accuracy, Macro-F1 on validation set
- **Model saved:** best_comment_classifier.pkl + label_encoder.pkl

### Prediction Pipeline :
- Converts input → DataFrame with exact column names
- Loads preprocessor.pkl → transforms features
- Loads model.pkl → predicts class
- Loads label_encoder.pkl → decodes to readable labels

### Flask App Creation :
- **Homepage** (`/`): Landing page
- **Prediction** (`/predictdata`): Form + results
- Responsive HTML/CSS templates
- Error handling with try-catch
- Real-time inference (<100ms)

## Exploratory Data Analysis Notebook

**Link:** [EDA Notebook](notebook\EDA Comment categ pred.ipynb)
## Model Training Notebook

**Link:** [Training Notebook](notebook/model_training.ipynb)

## Model Performance Results

| Input Example | Prediction | Confidence |
|---------------|------------|------------|
| "you are an idiot" | 2 (severe toxic) | 87% |
| "I will kill you" | 3 (threat) | 92% |
| "burned alive" | 0 (clean) | 91% |
| "Great post!" | 0 (clean) | 99% |

**Current Performance:** 79% Accuracy

## Future Improvements Planned

1. **Advanced TF-IDF** (char ngrams, trigrams): +3-5%
2. **GridSearchCV** hyperparameter tuning: +4-6%  
3. **Text features** (length, CAPS ratio): +5%
4. **Ensemble models** (XGBoost+Logistic): +2-3%

**Target Accuracy: 88-91%**



