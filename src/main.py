import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
import json
from Config import RAW_DATA_PATH, PROJECT_ROOT
from custon_classes import Missing_data_emulation
from data_preprocessing import prepare_base_data, grab_col_names


raw_df = pd.read_csv(RAW_DATA_PATH)
X, y = prepare_base_data(raw_df)

# Розділення на тренування, валідацію та тест
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val,y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) 

# Визначення списків назв стовпців за типом даних 
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

# Зчитування зафіксованих гіперпараметрів з JSON
params_path = PROJECT_ROOT / 'models' / 'best_params.json'
try:
    with open(params_path, 'r') as f:
        best_params = json.load(f)
    print(f" Успішно завантажено параметри з конфігу: {best_params}")
except FileNotFoundError:
    # Дефолтні підстрахувальні параметри, якщо файлу немає
    best_params = {'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 200}
    print(f" Файл конфігу не знайдено. Використовую дефолтні значення: {best_params}")

# Визначення диспропорції в класах
no_churn, churn = list(y_train.value_counts())
scale_weight = no_churn / churn

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

# Налаштування основної моделі 
model = xgb.XGBClassifier(
    **best_params, 
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