
# ## Problem Statement
# 
# The client/business deals with used cars sales.
# 
# The customers in this sector give strong preference to less-aged cars and popular brands with good resale value. This puts a very strong challenge as they only have a very limited range of vehicle options to showcase.

''' No Pre-Set Standards'''


''' How does one determine the value of a used car?
# The Market scenario is filled with a lot of malpractices. 
There is no defined standards exist to determine the appropriate price for the cars, 
the values are determined by arbitrary methods.
The unorganized and unstructured methods are disadvantageous to the both the 
parties trying to strike a deal. The look and feel can be altered in used cars, but 
the performance cannot be altered beyond a point.'''
 

''' Revolutionizing the Used Car Industry Through Machine Learning '''

# **Linear regression**
# Linear regression is a ML model that estimates the relationship between independent variables and a dependent variable using a linear equation (straight line equation) in a multidimensional space.

# **CRISP-ML(Q) process model describes six phases:**
# 
# - Business and Data Understanding
# - Data Preparation (Data Engineering)
# - Model Building (Machine Learning)
# - Model Evaluation and Tunning
# - Deployment
# - Monitoring and Maintenance
# 

''' Objective(s): Maximize the profits'''
# 
'''Constraints: Maximize the customer satisfaction'''

#**Success Criteria**
# - **Business Success Criteria**: Improve the profits from anywhere between 10% to 20%
# - **ML Success Criteria**: RMSE should be less than 0.15
# - **Economic Success Criteria**: Second/Used cars sales delars would see an increase in revenues by atleast 20%

# Importing necessary libraries

import pandas as pd
import seaborn as sb
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import pickle


cars = pd.read_csv(r"C:\Users\asus\DataScience\codes_DS_sample\project_deployment\Streamlit\app\Cars.csv")

cars

cars.isnull().any()

#### Descriptive Statistics and Data Distribution
cars.describe()

print(cars.corr())

dataplot = sb.heatmap(cars.corr(), annot = True, cmap = "YlGnBu")


# Seperating input and output variables 
X = cars.iloc[:, 1:6].values
y = cars.iloc[:, 0].values

y

X

# checking unique values
cars["Enginetype"].unique()

X.shape


# ### Define the steps for pipeline
ct = ColumnTransformer([("ODC", OrdinalEncoder(), [0])], remainder = "passthrough")
abc = ct.fit(X)

joblib.dump(abc, 'ordinalEnc')

final = abc.transform(X)

ordinal = OrdinalEncoder().fit(X)

ab = ordinal.transform(X)

columntranform1 = ct.fit(X)
q =  columntranform1.transform([['hybrid', 60, 60, 120, 33]])

q

# splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(final, y, test_size = 0.2, random_state = 0)


multilinear = LinearRegression()

multilinear.fit(X_train, y_train)

# Save the model
pickle.dump(multilinear, open('mlr.pkl','wb'))

# Load the saved model
model = pickle.load(open('mlr.pkl', 'rb'))

# Predicting upon X_test
y_pred = model.predict(X_test)
y_pred

# checking the Accurarcy by using r2_score
accuracy = r2_score(y_test, y_pred)

accuracy
