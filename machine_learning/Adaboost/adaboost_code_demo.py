import pandas as pd
import numpy as np
from mlxtend.plotting import plot_decision_regions


df = pd.DataFrame()

df['x1'] = [1,2,3,4,5,6,6,7,9,9]
df['x2'] = [5,3,6,8,1,9,5,8,9,2]
df['label'] = [1,1,0,1,0,1,0,1,0,0]


import seaborn as sns
sns.scatterplot(x=df['x1'], y=df['x2'],hue=df['label'])

# Here we assign the weight to each row

df['weight'] = 1/df.shape[0]

# Now here we creating the decison stumps
# Step = 1 

from sklearn.tree import DecisionTreeClassifier

dt1 = DecisionTreeClassifier(max_depth=1)
X = df.iloc[:,0:2].values
y = df.iloc[:,2].values

#Step 2 - train the first model

dt1.fit(X,y)

from sklearn.tree import plot_tree
plot_tree(dt1)

plot_decision_regions(X,y,clf=dt1, legend=2)

df['y_pred'] = dt1.predict(X)

df

# Step 3 = Finding the weight for the model

def calculate_alpha_weight(error):

    return 0.5*np.log((1-error)/error)

alpha1 = calculate_alpha_weight(0.3)
alpha1


# step 4 - update weight

def update_row_weights(row, alpha=0.423):
    if row['label'] ==row['y_pred']:
        return row['weight'] * np.exp(-alpha)
    else:
        return row['weight'] * np.exp(alpha)
    

df['updated_weight'] = df.apply(update_row_weights, axis=1)

df

df['updated_weight'].sum()



df['normalized_weight'] = df['updated_weight']/df['updated_weight'].sum()

df['normalized_weight'].sum()

df['cumsum_upper'] = np.cumsum(df['normalized_weight'])

df['cumsum_lower'] = df['cumsum_upper'] - df['normalized_weight']

df[['x1', 'x2',	'label', 'weight', 'y_pred', 'updated_weight', 'cumsum_lower','cumsum_upper']]

def create_new_dataset(df):
    
    indices = []

    for i in range(df.shape[0]):
        a = np.random.random()
        for index,row in df.iterrows():
            if row['cumsum_upper'] > a and a > row['cumsum_lower']:
                indices.append(index)
    return indices


index_values = create_new_dataset(df)

index_values

second_df = df.iloc[index_values,[0,1,2,3]]

print(second_df)

# Same process goes till you don't get more accurrate