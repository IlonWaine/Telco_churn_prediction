import pandas as pd

def grab_col_names(dataframe:pd.DataFrame, cat_th=10, car_th=20) -> tuple: 
    """
    Розподіляє колонки на категоріальні, числові та висококардинальні.
    Returns: tuple of (cat_cols, num_cols, cat_but_car)
    """
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

def prepare_base_data(raw_df: pd.DataFrame) -> tuple:
    """
    Базова підготовка таргету, видалення непотрібних колонок
    та очищення типів даних.
    """
    raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'],errors='coerce')

    # Визначення датасету фіч та таргету
    X = raw_df.drop(columns=['customerID', 'Churn']) 
    y = raw_df['Churn'].map({'Yes': 1, 'No': 0})
    return X, y