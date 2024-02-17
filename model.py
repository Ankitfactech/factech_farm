import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_excel('potato2.xlsx')
print(df.info())

plt.figure(figsize=(15,20))
sns.countplot(y=df['area'])
# plt.show()


country = df['area'].unique()
yield_per_country = []
for state in country:
    yield_per_country.append(df[df['area']==state]['yield'].sum())
plt.figure(figsize=(15, 20))
sns.barplot(y=country, x=yield_per_country)
# plt.show()

col = ['state', 'district','season','start', 'end', 'area', 'production']
df = df[col]
df = df.dropna()
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numeric_features = [3, 4, 5]
categorical_features = [0, 1, 2]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

clf = Pipeline(steps=[('preprocessor', preprocessor)])

clf.fit(X_train)
X_train_dummy = clf.transform(X_train)
X_test_dummy = clf.transform(X_test)

#linear regression
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor,RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from xgboost import XGBRegressor


models = {
    'knn':KNeighborsRegressor(),
    'etr':ExtraTreesRegressor(),
    'br':BaggingRegressor(),
    'ar':AdaBoostRegressor(),
    'xgb':XGBRegressor(),
    'lr':LinearRegression(),
    'rf':RandomForestRegressor(),
    'lss':Lasso(),
    'Rid':Ridge(),
    'Dtr':DecisionTreeRegressor(),
    'gbr':GradientBoostingRegressor()
}
for name, md in models.items():
    md.fit(X_train_dummy,y_train)
    y_pred = md.predict(X_test_dummy)
    
    print(f"{name} : mae : {mean_absolute_error(y_test,y_pred)} score : {r2_score(y_test,y_pred)}")

#just for testing..
xgb = XGBRegressor()
xgb.fit(X_train_dummy,y_train)
xg_pred=xgb.predict(X_test_dummy)

etr = ExtraTreesRegressor()
etr.fit(X_train_dummy,y_train)
etr_pred=etr.predict(X_test_dummy)

knn = KNeighborsRegressor()
knn.fit(X_train_dummy,y_train)
knn_pred=knn.predict(X_test_dummy)

br = BaggingRegressor()
br.fit(X_train_dummy,y_train)
br_pred=br.predict(X_test_dummy)

ar = AdaBoostRegressor()
ar.fit(X_train_dummy,y_train)
ar_pred=ar.predict(X_test_dummy)

lr = LinearRegression()
lr.fit(X_train_dummy,y_train)
lr_pred=lr.predict(X_test_dummy)

rf = RandomForestRegressor()
rf.fit(X_train_dummy,y_train)
rf_pred=rf.predict(X_test_dummy)

gbr = GradientBoostingRegressor()
gbr.fit(X_train_dummy,y_train)
gbr_pred=gbr.predict(X_test_dummy)

dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy,y_train)
dtr_pred=dtr.predict(X_test_dummy)


def prediction(state,district,season,start,end,area):
    # Create an array of the input features
    features = np.array([[state,district,season,start,end,area]], dtype=object)

    # Transform the features using the preprocessor
    transformed_features = preprocessor.transform(features)

    # Make the prediction
    predicted_yield = xgb.predict(transformed_features).reshape(1, -1)

    return predicted_yield[0]

state= 'Bihar'
district='GAYA'
season='Kharif'
start=2024
end =2025
area = 18
result = prediction(state,district,season,start,end,area)
print(result)

import joblib
joblib.dump(preprocessor,'./static/pre.joblib')
joblib.dump(knn,'./static/knn.joblib')
joblib.dump(etr,'./static/etr.joblib')
joblib.dump(br,'./static/br.joblib')
joblib.dump(ar,'./static/ar.joblib')
joblib.dump(xgb,'./static/xgb.joblib')
joblib.dump(lr,'./static/lr.joblib')
joblib.dump(rf,'./static/rf.joblib')
joblib.dump(gbr,'./static/gbr.joblib')
joblib.dump(dtr,'./static/dtr.joblib')

