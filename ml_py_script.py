# %%
import numpy as np
import pandas as pd

# %%
print("numpy=="+np.__version__)
print("pandas=="+pd.__version__)

# %%
df = pd.read_csv("/kaggle/input/datasets/sohaibdevv/how-creatine-and-mass-gainers-are-helping-people/supplement_impact_data.csv")

# %% [markdown]
# # Data Understanding

# %%
df.head()

# %%
df.tail()

# %%
df.sample(10)

# %%
df.shape

# %%
df.duplicated().sum()

# %%
df.isnull().sum()

# %%
df = df.drop(columns=["ID"])

# %%
df['difference_found_in_WT'] = df["Final_WT"]-df["Initial_WT"]
df = df.drop(columns=["Final_WT"])

# %%
df.head()

# %%
df.rename(columns={"Strength_Gain":"Strength_Gain(%)"},inplace=True)

# %%
df["Strength_Gain(%)"] = df["Strength_Gain(%)"].str.replace("%","").astype(int)

# %%
df.head()

# %%
df.dtypes

# %%
df.describe()

# %%
df.info()

# %%
df.select_dtypes(include=["int"]).drop(columns=["Strength_Gain(%)"]).columns

# %%
df.select_dtypes(include=["float"]).columns

# %%
df.select_dtypes(include=["object"]).columns

# %%
num_cols = df.select_dtypes(include=["int","float"]).drop(columns=["Strength_Gain(%)"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns
tar_col = df["Strength_Gain(%)"]

# %% [markdown]
# # Exploratory Data Analysis

# %%
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# %%
print("matplotlib=="+matplotlib.__version__)
print("seaborn=="+sns.__version__)

# %% [markdown]
# **Univariate Analysis**

# %% [markdown]
# *Numeric Columns*

# %%
right_skewed_cols = []
left_skewed_cols = []
nrml_num_cols = []
for col in num_cols:
    sns.kdeplot(x=col,data=df)
    plt.title(col)
    plt.show()
    print(df[col].skew())
    if df[col].skew()>=0.6:
        right_skewed_cols.append(col)
    elif df[col].skew()<=-0.6:
        left_skewed_cols.append(col)
    else:
        nrml_num_cols.append(col)

# %%
print(f"right skewed cols: {right_skewed_cols}")
print(f"left skewed cols: {left_skewed_cols}")
print(f"nrml skewed cols: {nrml_num_cols}")

# %% [markdown]
# * All numeric columns are almost perfect nrml skewed

# %%
for col in num_cols:
    sns.boxplot(x=col,data=df)
    plt.title(col)
    plt.show()

# %% [markdown]
# * No outliers found

# %% [markdown]
# *Categorical Columns*

# %%
for col in cat_cols:
    sns.countplot(y=col,data=df)
    plt.title(col)
    plt.show()

# %% [markdown]
# * Almost all are balanced

# %% [markdown]
# *Target Column*

# %%
sns.kdeplot(x=tar_col,data=df)
plt.title("Target Column KDE plot")
plt.show()
print(f"Target col skew: {df["Strength_Gain(%)"].skew()}")
sns.boxplot(x=tar_col,data=df)
plt.title("Target Column box plot")
plt.show()

# %% [markdown]
# Target column:
# 
# * Equal distribution, no skewness
# * Has no outliers

# %% [markdown]
# **Bivariate Analysis**

# %% [markdown]
# *Numeric Columns*

# %%
for col in num_cols:
    sns.regplot(x=col,y=tar_col,data=df,scatter=False)
    plt.title(col)
    plt.show()

# %% [markdown]
# * Age and weeks features are clearly increasing the Strength_Gain(%) value as they are increaseing 📈
# * Initial_WT: as it increases the target value decreases (almost straight line)
# * difference_found_in_WT: as it increases the target value increases (almost straight line)

# %% [markdown]
# *Categorical Columns*

# %%
for col in cat_cols:
    sns.violinplot(y=col,x=tar_col,data=df)
    plt.title(col)
    plt.show()

# %% [markdown]
# * Supplement feature is clearly seprating target value through its categories:
#    1. Mass Gainer --> range from approx 5-13 of Strength_Gain(%)
#    2. Creatine Monohydrate --> range from approx 7-28 of Strength_Gain(%)
#    3. Both --> range from approx 12-30 of Strength_Gain(%)
# * Gender and Primary_Benifit are showing  very less sepration

# %% [markdown]
# **Multivariate Analysis**

# %% [markdown]
# *Multicollinearity Detection*

# %%
sns.heatmap(df[num_cols].corr(),annot=True,fmt=".2f",cmap="Blues")

# %% [markdown]
# * There is no multicollinearity 

# %% [markdown]
# # Preprocessing

# %%
import sklearn
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,mean_absolute_error
import optuna

# %%
print("sklearn=="+sklearn.__version__)
print("xgboost=="+xgboost.__version__)
print("optuna=="+optuna.__version__)

# %%
df.sample(1)

# %%
x = df.drop(columns=["Strength_Gain(%)"])
y = df["Strength_Gain(%)"]

# %% [markdown]
# # Train Test Split

# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# %% [markdown]
# # Pipelines for every type of columns seprately

# %%
nrml_num_pipeline = Pipeline(steps=[
    ("num_imputer",SimpleImputer(strategy="mean")),
    ("scaling",StandardScaler())    
])

# %%
cat_pipeline = Pipeline(steps=[
    ("cat_imputer",SimpleImputer(strategy="most_frequent")),
    ("OHEncoder",OneHotEncoder(drop="if_binary",handle_unknown="ignore"))
])

# %% [markdown]
# # Preprocessing ColumnTransformer

# %%
preprocessing = ColumnTransformer(transformers=[
    ("nrml_num_pipeline",nrml_num_pipeline,nrml_num_cols),
    ("cat_pipeline",cat_pipeline,cat_cols)
],remainder="passthrough")

# %% [markdown]
# # Finding best model with best params (Optuna)

# %%
def objective(trial):
    model_name = trial.suggest_categorical("model", ["lr", "dt", "rf", "xgb"])

    if model_name == "lr":
        model = LinearRegression()

    elif model_name == "dt":
        model = DecisionTreeRegressor(
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10)
    )

    elif model_name == "rf":
        model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        n_jobs=-1
    )

    elif model_name == "xgb":
        model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 0, 5),
        reg_alpha=trial.suggest_float("reg_alpha", 0, 5),
        reg_lambda=trial.suggest_float("reg_lambda", 0, 5),
        n_jobs=-1,
        verbosity=0
    )

    pipeline = Pipeline(steps=[
        ("preprocessing",preprocessing),
        ("model",model)
    ])

    score = cross_val_score(pipeline,x_train,y_train,cv=5,scoring="neg_mean_squared_error")
    return score.mean()

# %%
study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=100)

# %%
params = study.best_params
params

# %%
model_name = params["model"]

if model_name == "lr":
    final_model = LinearRegression()

elif model_name == "dt":
    final_model = DecisionTreeRegressor(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"]
    )

elif model_name == "rf":
    final_model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        n_jobs=-1
    )

elif model_name == "xgb":
    final_model = XGBRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        n_jobs=-1,
        verbosity=0
    )

# %% [markdown]
# # Final Pipeline (preprocessing + best model with best params)

# %%
final_pipeline = Pipeline(steps=[
    ("preprocessing",preprocessing),
    ("final_model",final_model)
])

# %% [markdown]
# # Training 

# %%
final_pipeline.fit(x_train,y_train)

# %% [markdown]
# # Predicting on test data

# %%
y_train_pred = final_pipeline.predict(x_train)
y_test_pred = final_pipeline.predict(x_test)

# %% [markdown]
# # Model Evaluation

# %% [markdown]
# **Mean squared error**

# %%
train_mse = mean_squared_error(y_train,y_train_pred)
print(f"Train MSE: {train_mse}")
test_mse = mean_squared_error(y_test,y_test_pred)
print(f"Test MSE: {test_mse}")

# %% [markdown]
# **Root mean squared error**

# %%
train_rmse = root_mean_squared_error(y_train,y_train_pred)
print(f"Train RMSE: {train_rmse}")
test_rmse = root_mean_squared_error(y_test,y_test_pred)
print(f"Test RMSE: {test_rmse}")

# %% [markdown]
# **NRMSE std**

# %%
train_nrmse_std = train_rmse / y_train.std()
print(f"Train NRMSE std: {train_nrmse_std}")
test_nrmse_std = test_rmse / y_test.std()
print(f"Test NRMSE std: {test_nrmse_std}")

# %% [markdown]
# **NRMSE mean**

# %%
train_nrmse_mean = train_rmse / y_train.mean()
print(f"Train NRMSE mean: {train_nrmse_mean}")
test_nrmse_mean = test_rmse / y_test.mean()
print(f"Test NRMSE mean: {test_nrmse_mean}")

# %% [markdown]
# **NRMSE range**

# %%
train_nrmse_range = train_rmse / (y_train.max()-y_train.min())
print(f"Train NRMSE range: {train_nrmse_range}")
test_nrmse_range = test_rmse / (y_test.max()-y_test.min())
print(f"Test NRMSE range: {test_nrmse_range}")

# %% [markdown]
# **Mean absolute error**

# %%
train_mae = mean_absolute_error(y_train,y_train_pred)
print(f"Train MAE: {train_mae}")
test_mae = mean_absolute_error(y_test,y_test_pred)
print(f"Test MAE: {test_mae}")

# %% [markdown]
# **R2 score**

# %%
train_r2 = r2_score(y_train,y_train_pred)
print(f"Train r2: {train_r2}")
test_r2 = r2_score(y_test,y_test_pred)
print(f"Test r2: {test_r2}")

# %% [markdown]
# # Saving the final_pipeline

# %%
import joblib
joblib.dump(final_pipeline,"final_pipeline.pkl")

# %%
print("joblib=="+joblib.__version__)

# %% [markdown]
# # Model Explaination (Shap)

# %%
import shap

# %%
print("shap=="+shap.__version__)

# %%
shap_preprocessor = final_pipeline.named_steps["preprocessing"]
shap_model = final_pipeline.named_steps["final_model"]

# %%
feature_names = []
for col in shap_preprocessor.get_feature_names_out():
    feature_names.append(col.split("__")[-1])

# %%
x_train_t = pd.DataFrame(
    shap_preprocessor.transform(x_train),
    columns=feature_names
)

# %%
x_test_t = pd.DataFrame(
    shap_preprocessor.transform(x_test),
    columns=feature_names
)

# %%
explainer = shap.Explainer(shap_model,x_train_t)

# %%
shap_values = explainer(x_test_t)

# %%
shap.plots.bar(shap_values,max_display=len(feature_names))

# %%
shap.plots.beeswarm(shap_values,max_display=len(feature_names))

# %% [markdown]
# # Correlation Checking when all column values are in numbers

# %%
plt.figure(figsize=(10, 10))
combined_num_df = pd.concat([x_test_t, y_test.rename("target")], axis=1)
sns.heatmap(combined_num_df.corr(),annot=True,fmt=".2f",cmap="Blues")
plt.show

# %%



