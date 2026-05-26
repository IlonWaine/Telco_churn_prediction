import pandas as pd 
import xgboost as xgb
import json 

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from Config import RAW_DATA_PATH, PROJECT_ROOT
from data_preprocessing import grab_col_names, prepare_base_data

raw_df = pd.read_csv(RAW_DATA_PATH)
X, y = prepare_base_data(raw_df)

X_train, X_temp, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
cat_cols, num_cols, _ = grab_col_names(X_train)

# Перетворення числових даних
num_pipe = Pipeline([
    ('scaler', StandardScaler())
])

# Перетворення категорільних даних
cat_pipe = Pipeline([
    ('encoder',OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,encoded_missing_value=-2))   
])

# Повний процес перетворення даних
preprocessor = ColumnTransformer(
    transformers=[
        ('num',num_pipe,num_cols),
        ('cat',cat_pipe,cat_cols)
    ]
)

# Вказівка повертати pd.DataFrame
preprocessor.set_output(transform="pandas")

X_train_clean = preprocessor.fit_transform(X_train)

# Визначення диспропорції в класах
no_churn, churn = list(y_train.value_counts())
scale_weight = no_churn / churn

# Налаштування моделі для підбору параметрів
tune_model = xgb.XGBClassifier(
    tree_method="hist", 
    scale_pos_weight=scale_weight,
    eval_metric="auc",
    random_state=42
)

# Варіації параметрів
param_grid = {
    'n_estimators': [100, 450, 500, 550],       
    'max_depth': [3, 5],                  
    'learning_rate': [0.009, 0.01, 0.02,],             
}

# Налаштування GridSearchCv
grid_search = GridSearchCV(
    estimator=tune_model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3, 
    verbose=1,
    n_jobs=-1 
)

# Пошук найкращих параметрів
grid_search.fit(
    X_train_clean, y_train,
)

print(f"Найкращий ROC-AUC: {grid_search.best_score_:.4f}")
print("Найкращі параметри:", grid_search.best_params_)

# Автоматичне збереження результатів у JSON-файл
params_path = PROJECT_ROOT / 'models' / 'best_params.json'
with open(params_path, 'w') as f:
    json.dump(grid_search.best_params_, f, indent=4)

print(f"Найкращі параметри успішно збережено у: {params_path}")