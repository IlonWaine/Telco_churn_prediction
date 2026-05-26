import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
from Config import RAW_DATA_PATH, PROJECT_ROOT
from custon_classes import Missing_data_emulation


raw_df = pd.read_csv(RAW_DATA_PATH)

# returns (categorical_columns, numerical_columns, categorical_highy_cordinality_columns)
def grab_col_names(dataframe:pd.DataFrame, cat_th=10, car_th=20) -> tuple: 
    raw_cat_cols = dataframe.select_dtypes(include=['category', 'str', 'string']).columns.tolist()
    raw_num_cols = dataframe.select_dtypes(include=['number']).columns.tolist()
    # Пошук числових категоріальних назв стовпців
    num_but_cat = [col for col in raw_num_cols if dataframe[col].nunique() < cat_th]
    # Пошук категоріальних назв стовпців із високою кардинальністю
    cat_but_car = [col for col in raw_cat_cols if dataframe[col].nunique() > car_th]
    # Визначення всіх категоріальних назв стовпців виключивши високо кардинальні
    cat_cols = raw_cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    # Визначення числових назв стовпців
    num_cols = [col for col in raw_num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car


# Перетворення колонки в числову заповнюючи проблемні рядки NaN
raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'],errors='coerce')

# Визначення датасету фіч та таргету
X = raw_df.drop(columns=['customerID', 'Churn']) 
y = raw_df['Churn'].map({'Yes': 1, 'No': 0})

# Розділення на тренування, валідацію та тест
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val,y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)   

# Визначення диспропорції в класах
no_churn, churn = list(y_train.value_counts())
scale_weight = no_churn / churn

# Визначення списків назв стовпців за типом даних 
cat_cols, num_cols, cat_but_car = grab_col_names(X_train)

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

# Перетворення данних у належний вигляд
X_train_clean = preprocessor.fit_transform(X_train)
X_val_clean = preprocessor.transform(X_val)
X_test_clean = preprocessor.transform(X_test)

# Налаштування моделі для підбору параметрів
tune_model = xgb.XGBClassifier(
    tree_method="hist", 
    scale_pos_weight=scale_weight,
    eval_metric="auc",
    random_state=42
)

# Варіації параметрів
param_grid = {
    'n_estimators': [100, 200, 300, 500],       
    'max_depth': [3, 5],                  
    'learning_rate': [0.01, 0.02, 0.03],             
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

# Налаштування основної моделі 
model = xgb.XGBClassifier(
    **grid_search.best_params_, 
    tree_method="hist",
    scale_pos_weight=scale_weight,
    early_stopping_rounds=50,
    eval_metric="auc",
    random_state=42

)

# Фінальний pipeline
transformer = Pipeline([
    ('missing_dropout',Missing_data_emulation(missing_rate=0.15,random_state=42)),
    ('preprocessing',preprocessor),
    ('model',model)
])

# Навчання моделі
transformer.fit(
    X_train, y_train,
    model__eval_set=[(X_val_clean, y_val)],
    model__verbose=False
)

# Словник із результатами 
model_artifact = {
    'pipeline': transformer,
    'features': {'categorical':cat_cols, 'numerical':num_cols},
    'model_version': '1.0.0' 
}

# Вивід результатів
val_probabilities = transformer.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_probabilities)
val_accuracy = transformer.score(X_val, y_val)

print("\n====== РЕЗУЛЬТАТИ ФІНАЛЬНОГО ПАЙПЛАЙНУ ======")
print(f"ROC-AUC на валідації: {val_auc:.4f}")
print(f"Accuracy на валідації: {val_accuracy:.4f}")

# Збереження моделі на диск
joblib.dump(model_artifact, PROJECT_ROOT / 'models' / 'churn_predictor_pipeline.pkl')