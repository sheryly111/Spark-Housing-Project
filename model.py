# %% [markdown]
# # Exploration

# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('raw_merged.csv')


# %%
raw_data = raw_data.replace(['EX - Excellent'] , 'E - Excellent')
raw_data["case_created_date"] = raw_data["case_created_date"][:11]
raw_data["close_date"] = raw_data["close_date"][:11]
raw_data["last_case_update"] = raw_data["last_case_update"][:11]

#split data insto 70 train, 15 val, 15 test
temp_train_data, temp_data = train_test_split(raw_data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


print(raw_data['p_id'].value_counts())


print("Train data shape:", temp_train_data.shape)
print("Validation data shape:", val_data.shape)
print("Test data shape:", test_data.shape)

# %%
print(temp_train_data['overall_cond'].value_counts())
print('NA COUNTS: ', temp_train_data.isna().sum())
print('num sam ids: ', temp_train_data['sam_id'].nunique())


# %%
temp_train_data.head()
temp_train_data['overall_cond'].head()

# %% [markdown]
# # Additional Column(s)

# %%
'''def check(cond):
        if cond == 'F - Fair' or cond == 'P - Poor':
            return 1
        else:
            return 0
train_data['prob_address'] = train_data['overall_cond'].apply(check)


print(train_data['overall_cond'].value_counts())
print(train_data['prob_address'].value_counts())'''

problematic_values = ['F - Fair', 'P - Poor']
temp_train_data[['overall_cond','ext_cond','int_cond']] = temp_train_data[['overall_cond','ext_cond','int_cond']].fillna('')

temp_train_data['prob_address'] = (
    temp_train_data['overall_cond'].str.strip().isin(problematic_values) |
    temp_train_data['ext_cond'].str.strip().isin(problematic_values) |
    temp_train_data['int_cond'].str.strip().isin(problematic_values)
).astype(int)
print(temp_train_data['prob_address'].value_counts())

# %% [markdown]
# # TRAIN TEST SPLIT!!!

# %%
from sklearn.utils import resample


sample_size = 0.99  # Adjust sample size as needed
print("Taking smaller sample size for faster processing...")

#train_data_with_target = train_data[train_data["prob_address"].notna()].copy()

from sklearn.model_selection import train_test_split
_, train_data_sample = train_test_split(
    temp_train_data, #split the train data again so that it is easier to test our feature engineering 
    test_size = sample_size, 
    random_state=42, 
    #stratify=train_data_with_target["prob_address"] 
)






print("Sampled training set shape: ", train_data_sample.shape)
print("Sampled prob distribution", train_data_sample["prob_address"].value_counts())



# %% [markdown]
# # Feature Engineering 

# %%
import numpy as np
import datetime 
from datetime import date
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD 
import pickle 

#feature engineering function
def create_features(dataset):

    #Feature 0: Fix Values 
    
    # Fill in Nans 
    print("filling in nans...")
    dataset["case_title"] = dataset["case_title"].fillna("").astype(str)
    dataset["case_subject"] = dataset["case_subject"].fillna("").astype(str)
    dataset["case_reason"] = dataset["case_reason"].fillna("").astype(str)
    dataset["over_5"] = dataset["over_5"].fillna("").astype(str)
    dataset["int_cond"] = dataset["int_cond"].fillna("").astype(str)
    dataset["ext_cond"] = dataset["ext_cond"].fillna("").astype(str)
    dataset["overall_cond"] = dataset["overall_cond"].fillna("").astype(str)
    dataset["bdrm_cond"] = dataset["bdrm_cond"].fillna("").astype(str)
    dataset["heat_type"] = dataset["heat_type"].fillna("").astype(str)
    dataset["ac_type"] = dataset["ac_type"].fillna("").astype(str)
    dataset["parcel_num"] = dataset["parcel_num"].astype(str)
    dataset["cm_id"] = dataset["cm_id"].astype(str)

    dataset["close_date"] = dataset["close_date"].fillna(date.today())
    dataset["close_date"] = dataset["close_date"].astype(str)
    
    # String -> Integer 
    dataset["year"] = dataset["year"].astype(int)

    # Standardize Values
    dataset["over_5"] = dataset["over_5"].replace(["N", "No"], "NO")
    dataset["over_5"] = dataset["over_5"].replace(["Y"], "YES")
    dataset["over_5"] = dataset["over_5"].replace(["nan"], "")
    
    #Feature 1: Text Features 
    print("Extracting text features...")
    dataset["case_reason_length"] = dataset["case_reason"].fillna('').astype(str).str.len()
    dataset['case_reason_word_count'] = dataset['case_reason'].fillna('').astype(str).str.split().str.len()

    dataset['case_title_length'] = dataset['case_title'].fillna('').astype(str).str.len()
    dataset['case_title_word_count'] = dataset['case_title'].fillna('').astype(str).str.split().str.len()

    dataset['case_type_length'] = dataset['case_type'].fillna('').astype(str).str.len()
    dataset['case_type_word_count'] = dataset['case_type'].fillna('').astype(str).str.split().str.len()

    #Feature 2: Time Based Features 
    print("doing time based features...")
    dataset['case_date'] = pd.to_datetime(dataset['case_created_date'], errors='coerce', infer_datetime_format=True)
    dataset['case_closing_date'] = pd.to_datetime(dataset['close_date'], errors='coerce', infer_datetime_format=True)

    #dataset['case_date'] = pd.to_datetime(dataset['case_created_date'], unit='s')
    dataset['case_year'] = dataset['case_date'].dt.year
    dataset['case_month'] = dataset['case_date'].dt.month
    dataset['case_day_of_week'] = dataset['case_date'].dt.dayofweek

    #dataset['case_closing_date'] = pd.to_datetime(dataset['close_date'])
    dataset['close_year'] = dataset['case_closing_date'].dt.year
    dataset['close_month'] = dataset['case_closing_date'].dt.month
    dataset['close_day_of_week'] = dataset['case_closing_date'].dt.dayofweek


    # Feature 3: Cyclical encoding for month and day of week
    print("Cyclical encoding for month and day...")
    dataset['created_month_sin'] = np.sin(2 * np.pi * dataset['case_month'] / 12)
    dataset['created_month_cos'] = np.cos(2 * np.pi * dataset['case_month'] / 12)
    dataset['created_dow_sin'] = np.sin(2 * np.pi * dataset['case_day_of_week'] / 7)
    dataset['created_dow_cos'] = np.cos(2 * np.pi * dataset['case_day_of_week'] / 7)

    dataset['closed_month_sin'] = np.sin(2 * np.pi * dataset['close_month'] / 12)
    dataset['closed_month_cos'] = np.cos(2 * np.pi * dataset['close_month'] / 12)
    dataset['closed_dow_sin'] = np.sin(2 * np.pi * dataset['close_day_of_week'] / 7)
    dataset['closed_dow_cos'] = np.cos(2 * np.pi * dataset['close_day_of_week'] / 7)

    # Feature 3.1: Length of case open
    dataset['case_open_days'] = (dataset['case_closing_date'] - dataset['case_date']).dt.days
    dataset["avg_resolution_time"] = dataset.groupby("full_address")["case_open_days"].transform("mean")

    # Feature 3.2: Num cases per address
    print("calculating cases per address per year...")
    cases_per_address_year = (
        dataset.groupby(["full_address", "year"])["case_enquiry_id"]
            .nunique()
            .reset_index(name="case_count")
        )

    dataset = dataset.merge(  # your base dataset of all addresses
        cases_per_address_year,
        on=["full_address", "year"],
        how="left"
    )

    # Feature 4: Severity Analysis
    print("performing sentiment intensity analysis...")
    sia = SentimentIntensityAnalyzer()
    dataset["case_reason_severity_score"] = dataset["case_reason"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
    dataset["case_title_severity_score"] = dataset["case_title"].apply(lambda x: sia.polarity_scores(str(x))["compound"])
    

    # Feature 6: Frequency of Cases
    print("finding frequency of cases...")
    dataset["years_with_cases"] = dataset.groupby("full_address")["year"].transform("nunique")
    dataset["total_case_count_address"] = dataset.groupby("full_address")["case_enquiry_id"].transform("count")
    dataset["avg_cases_per_year"] = dataset["total_case_count_address"] / dataset["years_with_cases"]

    # Feature 7: Escalation of Severity
    print("finding escalation of severity... ")
    '''
    year_counts = dataset.groupby(["full_address", "year"])["case_enquiry_id"].count()
    dataset["year_over_year_increase"] = year_counts.groupby(level=0).diff().fillna(0)
    ''' 
    # Feature 7: Escalation of Severity
    print("finding escalation of severity... ")
    counts_df = (
        dataset.groupby(["full_address", "year"], as_index=False)["case_enquiry_id"]
            .count()
            .rename(columns={"case_enquiry_id": "case_count"})
    )
    counts_df = counts_df.sort_values(["full_address", "year"])
    counts_df["year_over_year_increase"] = counts_df.groupby("full_address")["case_count"].diff().fillna(0)

    # Merge results back to dataset
    dataset = dataset.merge(counts_df[["full_address", "year", "case_count", "year_over_year_increase"]],
                            on=["full_address", "year"],
                            how="left")



    # Feature 8: Repeat closure failure / missed deadlines
    print("finding if metdeadline...")
    dataset['met_deadline'] = (dataset['case_met_deadline'] == "YES").astype(int)
    dataset['deadline_fail_rate'] = dataset.groupby("full_address")["met_deadline"].transform(lambda x: 1 - x.mean())

    # Feature 9: Condition Flags
    print("mapping conditions to scores ")
    condition_map = {
        "E - Excellent" : 1,
        "VG - Very Good" : 2,
        "G - Good" : 3,  
        "A - Average": 4,
        "F - Fair" : 5, 
        "P - Poor" : 6  
    }

    dataset["overall_cond_score"] = dataset["overall_cond"].map(condition_map).fillna(0)
    dataset["int_cond_score"] = dataset["int_cond"].map(condition_map).fillna(0)
    dataset["ext_cond_score"] = dataset["ext_cond"].map(condition_map).fillna(0)

    dataset["avg_overall_condition"] = dataset.groupby("full_address")["overall_cond_score"].transform("mean")
    dataset['case_date'] = dataset['case_date'].astype('int64') // 10**9
    dataset['case_closing_date'] = dataset['case_closing_date'].astype('int64') // 10**9

    #log transformations
    print("doing log transformations...")
    dataset['log_case_reason_length'] = np.log1p(dataset['case_reason_length'])
    dataset['log_case_reason_word_count'] = np.log1p(dataset['case_reason_word_count'])
    dataset['log_case_title_length'] = np.log1p(dataset['case_title_length'])
    dataset['log_case_title_word_count'] = np.log1p(dataset['case_title_word_count'])
    dataset['log_case_type_length'] = np.log1p(dataset['case_type_length'])
    dataset['log_case_type_word_count'] = np.log1p(dataset['case_type_word_count'])

    

    return dataset

    
    


# %% [markdown]
# # Apply Feature Engineering

# %%
#apply feature engineering 

dataset = train_data_sample.copy() 

dataset_features = create_features(dataset) 

########################################
# UPSAMPLING MINORITY CLASSES
########################################

train_majority = dataset_features[dataset_features['prob_address'] == 1]
train_minority_1 = dataset_features[dataset_features['prob_address'] == 0]

# Target number of samples = majority class
n_target = len(train_majority)

train_minority_1_upsampled = resample(train_minority_1, replace=True, n_samples=n_target, random_state=42)

# combine back
dataset_features = pd.concat([
    train_majority,
    train_minority_1_upsampled
], ignore_index=True)

# shuffle
dataset_features = dataset_features.sample(frac=1, random_state=42).reset_index(drop=True)

print("After upsampling, train Score distribution:\n", dataset_features['prob_address'].value_counts().sort_index())


# Feature 5: TF-IDF
print("doing tfidf on title and case reasons...")
# For case reason: 
case_reason_tfidf = TfidfVectorizer(
    max_features=2000,    # adjust based on memory / dataset size --> 4k 
    ngram_range=(1, 2),   # unigrams + bigrams
    stop_words='english',
    min_df=10,             
    max_df=0.8,
    sublinear_tf=True           
)

case_reason_tfidf_transformed = case_reason_tfidf.fit_transform(dataset_features['case_reason']) 

# For case title 
case_title_tfidf = TfidfVectorizer(
    max_features=3000,    # adjust based on memory / dataset size --> 4k 
    ngram_range=(1, 2),   # unigrams + bigrams
    stop_words='english',
    min_df=10,             
    max_df=0.8,
    sublinear_tf=True           
)

case_title_tfidf_transformed = case_title_tfidf.fit_transform(dataset_features['case_title']) 
with open ('case_reason_tfidf.pkl', 'wb') as f:
    pickle.dump(case_reason_tfidf, f)
with open ('case_title_tfidf.pkl', 'wb') as f:
    pickle.dump(case_title_tfidf, f)

# Feature 5.1: SVD 
print("truncate svd-ing...")
svd = TruncatedSVD(n_components=5, random_state=42)
case_reason_svd = svd.fit_transform(case_reason_tfidf_transformed)
case_title_svd = svd.fit_transform(case_title_tfidf_transformed)

for i in range(case_reason_svd.shape[1]):
    dataset_features[f'reason_tfidf_svd{i}'] = case_reason_svd[:,i]

for i in range(case_title_svd.shape[1]):
    dataset_features[f'title_tfidf_svd{i}'] = case_title_svd[:,i]

with open ('case_reason_svd.pkl', 'wb') as f:
    pickle.dump(case_reason_svd, f)
with open ('case_title_svd.pkl', 'wb') as f:
    pickle.dump(case_title_svd, f)



#drop non numerical features
print("dropping non numerical features...")
nonnumerical_cols = dataset_features.select_dtypes(include=['object']).columns
print("num non numerical columns to drop: ", len(nonnumerical_cols))
dataset_features = dataset_features.drop(columns=nonnumerical_cols)

additional_cols_to_drop = ['sam_id','building_id', 'ward_id', 'case_enquiry_id', 'gis_id' ]
dataset_features = dataset_features.drop(columns=additional_cols_to_drop)





# %% [markdown]
# # Train/Test Split

# %%
from sklearn.preprocessing import StandardScaler

print("Splitting data into train/validation sets BEFORE calculating ...")
print(dataset_features.shape)
df_filled = dataset_features.fillna(0)
data_for_modeling = df_filled[dataset_features['prob_address'].notna()].copy()
data_for_modeling['prob_address'] = data_for_modeling['prob_address'].astype(int)

print("Data for modeling shape:", data_for_modeling.shape)
# Split into train and validation sets (keeping original indices for mapping)
train_data, val_data = train_test_split(
    data_for_modeling,
    test_size=0.5,  # 50% for validation
    random_state=42,
    stratify=data_for_modeling['prob_address']
)



print(f"Training set shape: {train_data.shape}")
print(f"Validation set shape: {val_data.shape}")
print(f"Train prob distribution:\n{train_data['prob_address'].value_counts().sort_index()}")



# Prepare features for modeling (these will be used for final model training)
feature_columns = [col for col in train_data.columns if col not in ['prob_address']]
X_train_unscaled = train_data[feature_columns].copy()
y_train = train_data['prob_address'].copy()
X_val_unscaled = val_data[feature_columns].copy()
y_val = val_data['prob_address'].copy()


# print(f"\nFeatures shape - Train: {X_train_unscaled.shape}, Val: {X_val_unscaled.shape}")
# print(f"Feature columns ({len(feature_columns)}): {feature_columns}")


# # Standard Scaling (important for KNN and distance-based algorithms)
# # Scaling is applied AFTER log transformation to:
# # 1. Handle features with different scales (log transforms help, but scaling normalizes further)
# # 2. Ensure all features contribute equally to distance calculations in KNN
# # 3. Improve convergence and model performance
# print("\n" + "="*60)
# print("Applying Standard Scaling to features...")
# print("="*60)
# print("Standard Scaling normalizes features to have:")
# print("  - Mean = 0")
# print("  - Standard deviation = 1")
# print("\nNote: We fit the scaler on training data only, then transform train, val, and test")
# print("to avoid data leakage.")

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_unscaled)
# X_val_scaled = scaler.transform(X_val_unscaled)

# # Save scaler
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)

# print(f"\nTraining set shape (scaled): {X_train_scaled.shape}")
# print(f"Validation set shape (scaled): {X_val_scaled.shape}")






# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_unscaled, y_train)

print("Logistic Regression model trained with 1000 max iterations")

with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved as logistic_regression_model.pkl ')

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("Feature columns saved to 'feature_columns.pkl'")



# %%
import matplotlib.pyplot as plt
model.coef_[0]
# Create a DataFrame to view feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_[0],
    'Absolute_Coefficient': abs(model.coef_[0])
})

# Sort by absolute coefficient to see most important features
feature_importance_df = feature_importance_df.sort_values(by='Absolute_Coefficient', ascending=False)


print(feature_importance_df)

# top_n = 20
# top_features = feature_importance_df.head(top_n)
# plt.figure(figsize=(14, 12))
# plt.barh(top_features['Feature'], top_features['Coefficient'], color='skyblue')
# plt.xlabel("Coefficient Value")
# plt.title("Top 20 Most Important Features")
# plt.gca().invert_yaxis()
# plt.tight_layout()
#plt.show()





# %%
y_train_pred = model.predict(X_train_unscaled)
y_val_pred = model.predict(X_val_unscaled)

print("Training Predictions:", np.count_nonzero(y_train_pred))
print("Validation Predictions:", np.count_nonzero(y_train_pred))

# %% [markdown]
# # Test Set Evaluation
# 

# %%
# Evaluation
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report


# Predict on training and validation sets
y_train_pred = model.predict(X_train_unscaled)
y_val_pred = model.predict(X_val_unscaled)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

train_f1 = f1_score(y_train, y_train_pred, average='weighted')
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

# Per-class F1 scores
per_class_f1 = f1_score(y_val, y_val_pred, average=None, labels=[1,0])

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"\nTraining Set:")
print(f"  Accuracy: {train_accuracy:.4f}")
print(f"  F1 Score (weighted): {train_f1:.4f}")

print(f"\nValidation Set:")
print(f"  Accuracy: {val_accuracy:.4f}")
print(f"  F1 Score (weighted): {val_f1:.4f}")

print(f"\nPer-class F1 Scores:")
for score, label in zip(per_class_f1, [1,0]):
    print(f"  Score {label}: {score:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, labels=[1,0]))

# Plot confusion matrix
cm = confusion_matrix(y_val, y_val_pred, labels=[1,0])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[1,0], yticklabels=[1,0])
plt.title('Confusion Matrix - Validation Set')
plt.xlabel('Predicted Score')
plt.ylabel('True Score')
plt.show()

print("\nEvaluation complete!")

# %% [markdown]
# # Test Set Eval

# %%

# Create predictions for test.csv
print("Processing test set for predictions...")

test_dataset = test_data.copy()
test_dataset_features = create_features(test_dataset)


# Load trained vectorizer and SVD from training phase
with open('case_reason_tfidf.pkl', 'rb') as f:
    case_reason_tfidf = pickle.load(f)
with open('case_title_tfidf.pkl', 'rb') as f:
    case_title_tfidf = pickle.load(f)

# Feature 5: TF-IDF
print("doing tfidf on title and case reasons...")


case_reason_tfidf_transformed = case_reason_tfidf.fit_transform(test_dataset_features['case_reason']) 


case_title_tfidf_transformed = case_title_tfidf.fit_transform(test_dataset_features['case_title']) 




with open('case_title_svd.pkl', 'rb') as f:
    case_title_svd = pickle.load(f)
with open('case_reason_svd.pkl', 'rb') as f:
    case_reason_svd = pickle.load(f)

# Feature 5.1: SVD 
print("truncate svd-ing...")

case_reason_svd = svd.fit_transform(case_reason_tfidf_transformed)
case_title_svd = svd.fit_transform(case_title_tfidf_transformed)

for i in range(case_reason_svd.shape[1]):
    test_dataset_features[f'reason_tfidf_svd{i}'] = case_reason_svd[:,i]

for i in range(case_title_svd.shape[1]):
    test_dataset_features[f'title_tfidf_svd{i}'] = case_title_svd[:,i]


#drop non numerical features
print("dropping non numerical features...")
nonnumerical_cols = test_dataset_features.select_dtypes(include=['object']).columns
print("num non numerical columns to drop: ", len(nonnumerical_cols))
test_dataset_features = test_dataset_features.drop(columns=nonnumerical_cols)

additional_cols_to_drop = ['sam_id','building_id', 'ward_id', 'case_enquiry_id', 'gis_id' ]
test_dataset_features = test_dataset_features.drop(columns=additional_cols_to_drop)


X_test_features = test_dataset_features[feature_columns].copy()
X_test_features.fillna(0, inplace=True)
test_predictions = model.predict(X_test_features)

# # Prepare features for modeling (these will be used for final model training)
# feature_columns = [col for col in test_data.columns if col not in ['prob_address']]
# X_test_unscaled = test_data[feature_columns].copy()




# %% [markdown]
# # Landlord Analysis

# %%
test_complete = X_test_features.copy()
test_complete['prob_address'] = test_predictions

#print(test_complete.nunique())
num_prob_per = (
  
    test_complete.groupby("p_id")["prob_address"]
      .sum()
      .reset_index(name="num_prob_per")
)

#test_complete = test_complete.merge(num_prob_per, on="p_id", how="left")

# test_complete["num_prob_per"] = (
#     test_complete.groupby("p_id")["prob_address"].transform("sum")
# )

#num_prob_per = test_complete.groupby("p_id")["prob_address"].sum()
num_prob_per.head()
print(num_prob_per['num_prob_per'].shape)
print(test_complete['p_id'].unique().shape)

test_complete = test_complete.merge(num_prob_per, on="p_id", how="left")

#print(test_complete['num_prob_per'].value_counts())
print(test_complete['p_id'].value_counts())
#print(test_complete['num_prob_per'].value_counts())





# %% [markdown]
# # Additional Visualizations

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Group by year and take average predicted risk
risk_over_time = test_complete.groupby("year")["prob_address"].mean().reset_index()
risk_over_time['rolling_avg'] = risk_over_time['prob_address'].rolling(3, center=True).mean()


plt.figure(figsize=(10, 6))
plt.hist(test_complete['prob_address'], bins=2, color='#d9cfe6', edgecolor='white')
plt.title("Distribution of Predicted Flags")
plt.xlabel("Predicted Flag (0 = Not Problematic, 1 = Problematic)")
plt.ylabel("Count of Properties")
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(risk_over_time['year'], risk_over_time['prob_address'], marker='o')
plt.title("Average Predicted Risk per Year")
plt.xlabel("Year")
plt.ylabel("Average Predicted Probability of Bad Property")
plt.grid(True)
plt.xticks(risk_over_time['year'])  # ensure all years are labeled
plt.show()

print(test_complete.columns)




# %% [markdown]
# 

# %% [markdown]
# # Build New Dataframe for Interpretability

# %%
print(test_complete.columns)

# %%
landlord_predictions = test_complete[['p_id', 'num_prob_per']].copy()

print(landlord_predictions.shape)
print(landlord_predictions['p_id'].nunique())

print(landlord_predictions['p_id'].value_counts())


#find num total properties per landlord
ordered_prop = landlord_predictions['p_id'].unique()
prop_counts = landlord_predictions['p_id'].value_counts().reindex(ordered_prop)


landlord_predictions = landlord_predictions.drop_duplicates().reset_index(drop=True)

landlord_predictions['total_properties'] = prop_counts.values

landlord_predictions.head()

#landlord_predictions.head()


# %%
landlord_predictions["percentage_prob"] = landlord_predictions["num_prob_per"] / landlord_predictions["total_properties"]
landlord_predictions.drop(landlord_predictions[landlord_predictions["total_properties"] > 400].index, inplace=True)
landlord_predictions.head()

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(landlord_predictions["total_properties"], landlord_predictions["num_prob_per"])
plt.xlabel("Total Properties Owned")
plt.ylabel("num Problematic")
plt.title("Do Larger Landlords Have More Problems!")
plt.grid(True)
plt.show()


# %% [markdown]
# # Identifying Problematic Landlords

# %%
landlord_predictions["problematic?"] = landlord_predictions["percentage_prob"] > 0.5
landlord_predictions.head()
print("total number of landlords:", landlord_predictions.shape[0])
print("number of predicted problematic landlords:", landlord_predictions["problematic?"].sum())

# %% [markdown]
# # Save CSVs

# %%
test_complete.to_csv('test_set_predictions.csv', index=False)
landlord_predictions.to_csv('landlord_predictions.csv', index=False) 

# %%



