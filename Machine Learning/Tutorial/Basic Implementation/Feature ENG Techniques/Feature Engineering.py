import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


filename = ""
dataset = pd.read_csv(filename)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1:].values

##################### SPLIT #####################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)







##################### SCALE SCALING #####################
# Standard Scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#  Min Max Scaler
from sklearn.preprocessing import MinMaxScaler
MM_scaler = MinMaxScaler()
X_train = MM_scaler.fit_transform(X_train)
X_test = MM_scaler .transform(X_test)

# Power Transformer
from sklearn.preprocessing import PowerTransformer
pow_trans = PowerTransformer()
X_train = pow_trans.fit_transform(X_train)
X_test = pow_trans .transform(X_test)








##################### Categorical one hot dummy label encoding #####################

dummy_encoding = pd.get_dummies(my_dataframe, columns=[''], prefix='')

one_hot_encoding = pd.get_dummies(my_dataframe, columns=[''], prefix='', drop_first=True)







##################### Binary Binarizing data columns ##################

my_dataframe["newColumn"] = 0
threshold = 0
# Replace all the values where myColumn is > threshold
my_dataframe.newColumn[my_dataframe[my_dataframe["myColumn"] > threshold].index] = 1








#################### Binning categorizing dividing splitting data to equal groups ############
my_dataframe['equal_binned'] = pd.cut(my_dataframe['myColumn'], bins=5)







#################### Drop fill missing nan values ################
# Remove entire row or column
no_missing_values_rows = my_dataframe.dropna(how="any", axis=0) # 0 removes rows | 1 removes columns

# remove row/col in specific variable
no_missing_values = my_dataframe.dropna(subset=["column"])

# fill missing category values
my_dataframe['column'].fillna("Not Given", inplace=True)

# fill missing numeric values
my_dataframe['column'].fillna(round(my_dataframe['column'].mean()), inplace=True)

#Detect characthers that prevent string data to be converted to digits
def getStrangeChars(my_dataframe, column):
    numeric_vals = pd.to_numeric(my_dataframe[column], errors='coerce')
    # Find the indexes of missing values
    idx = numeric_vals.isnull()
    # Print the relevant rows
    return my_dataframe[column].loc[idx]


# Replace modify string
my_dataframe['column'] = my_dataframe['column'].str.replace("$","")








################## Visualization ##############
# Histogram
my_dataframe.hist()
plt.show()

# Boxplot
# Create a boxplot of two columns
my_dataframe[['column1', 'column2']].boxplot()
plt.show()

# pairplot pairwise relationships
sns.pairplot(my_dataframe)
plt.show()






################ Remove drop outliers #############
# Remove data >= quantile 95%
quantile = my_dataframe['column'].quantile(0.95)
trimmed_df = my_dataframe[my_dataframe['column'] < quantile]
# The trimmed histogram
trimmed_df[['column']].hist()
plt.show()


# Statistical outlier removal
# Find the mean and standard dev
std = my_dataframe['column'].std()
mean = my_dataframe['column'].mean()
# Calculate the cutoff
cut_off = std * 3
lower, upper = mean - cut_off, mean+cut_off
# Trim the outliers
trimmed_df = my_dataframe[(my_dataframe['column'] < upper) 
                           & (my_dataframe['column'] > lower)]







############## String text clean ###############
# Replace all non letter characters with a whitespace
my_dataframe['text_clean'] = my_dataframe['text'].str.replace('[^a-zA-Z]', ' ')
# Change to lower case
my_dataframe['text_clean'] = my_dataframe['text_clean'].str.lower()


# Words count
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv_transformed = cv.fit_transform(my_dataframe['text_clean'])
cv_array = cv_transformed.toarray()


# CountVectorizer to Dataframe
cv_df = pd.DataFrame(cv_array, 
                     columns=cv.get_feature_names())
# Add the new columns to the original DataFrame
my_dataframe_new = pd.concat([my_dataframe, cv_df], axis=1, sort=False)


# Term frequency-inverse document frequency (Tf idf)
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features=100, stop_words="english")
tv_transformed = tv.fit_transform(my_dataframe['text_clean'])
# Create a DataFrame with these features
tv_df = pd.DataFrame(tv_transformed.toarray(), 
                     columns=tv.get_feature_names())

# Ngram N-gram n gram
cv_trigram_vec = CountVectorizer(max_features=100, 
                                 stop_words='english', 
                                 ngram_range=(2,2))

# Most/top common words
# Create a DataFrame of the features
cv_tri_df = pd.DataFrame(cv_transformed.toarray(), 
                 columns=cv.get_feature_names())
# Print the top 5 words in the sorted output
print(cv_tri_df.sum().sort_values(ascending=False).head(5))