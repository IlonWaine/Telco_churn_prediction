import pandas as pd
import numpy as np

# Базові набори значень для текстових ознак
categorical_options = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    "SeniorCitizen": [0, 1]
}

num_rows = 100

# =====================================================================
# FILE 1: ГАРНІ ДАНІ (test_good_data.csv)
# Абсолютно ідеальні рядки, без жодного пропуску, з адекватними числами
# =====================================================================
np.random.seed(42) # фіксуємо результат для відтворюваності

good_data = {}
for col, options in categorical_options.items():
    good_data[col] = np.random.choice(options, size=num_rows)

# Генеруємо реалістичні числа
good_data["customerID"] = [f"{np.random.randint(1000, 9999)}-{ ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=4)) }" for _ in range(num_rows)]
good_data["tenure"] = np.random.randint(1, 73, size=num_rows) # від 1 до 72 місяців
good_data["MonthlyCharges"] = np.round(np.random.uniform(18.25, 118.75, size=num_rows), 2)
good_data["TotalCharges"] = np.round(good_data["tenure"] * good_data["MonthlyCharges"], 2)

df_good = pd.DataFrame(good_data)
df_good.to_csv("test_good_data.csv", index=False)
print("✅ Файл 'test_good_data.csv' успішно створено (100 рядків).")


# =====================================================================
# FILE 2: ЕКСТРЕМАЛЬНІ ДАНІ ТА ПРОПУСКИ (test_extreme_data.csv)
# Містить аномальні викиди, невідомі категорії, пропуски (NaN/None)
# =====================================================================
np.random.seed(24)

extreme_data = {}
for col, options in categorical_options.items():
    extreme_data[col] = np.random.choice(options, size=num_rows)

good_data["customerID"] = [f"{np.random.randint(1000, 9999)}-{ ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=4)) }" for _ in range(num_rows)]

# 1. Додаємо екстремальні викиди у числа
extreme_data["tenure"] = np.random.randint(1, 73, size=num_rows)
extreme_data["tenure"][10] = 0        # Клієнт підключився сьогодні (0 місяців)
extreme_data["tenure"][20] = 500      # Лояльність 40 років (аномальний викид)

extreme_data["MonthlyCharges"] = np.round(np.random.uniform(18.25, 118.75, size=num_rows), 2)
extreme_data["MonthlyCharges"][30] = 9999.99  # Скажений щомісячний чек
extreme_data["MonthlyCharges"][40] = -50.00    # Від'ємна вартість

extreme_data["TotalCharges"] = str(np.round(extreme_data["tenure"] * extreme_data["MonthlyCharges"], 2))

# 2. Перетворюємо масиви у DataFrame, щоб закинути туди «бруд»
df_extreme = pd.DataFrame(extreme_data)

# 3. Штучно створюємо пропуски (NaN) в різних колонках
# Наш фронтенд і бекенд мають їх "проковтнути" завдяки imputer у пайплайні
df_extreme.loc[df_extreme.sample(frac=0.15).index, 'TotalCharges'] = np.nan
df_extreme.loc[df_extreme.sample(frac=0.10).index, 'tenure'] = np.nan
df_extreme.loc[df_extreme.sample(frac=0.10).index, 'InternetService'] = np.nan
df_extreme.loc[df_extreme.sample(frac=0.05).index, 'PaymentMethod'] = np.nan

# 4. Додаємо «зламані» або непередбачувані текстові значення (тест на стійкість кодування)
df_extreme.loc[15, 'gender'] = "Unknown"  # Нова категорія, якої не було при тренуванні
df_extreme.loc[25, 'Contract'] = "3 years" # Нетиповий контракт
df_extreme.loc[35, 'TotalCharges'] = " "   # Пробіл замість числа (класика Telco датасету)

df_extreme.to_csv("test_extreme_data.csv", index=False)
print("⚠️ Файл 'test_extreme_data.csv' успішно створено (100 рядків із викидами та пропусками).")