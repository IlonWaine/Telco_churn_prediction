import joblib
import pandas as pd
from Config import PROJECT_ROOT
from pydantic import create_model
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
from contextlib import asynccontextmanager


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

def get_html_response(file_name: str) -> HTMLResponse:
    with open(PROJECT_ROOT / 'templates' / file_name, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.get("/", response_class=HTMLResponse)
async def home_page():
    return get_html_response("home.html")


@app.get("/predict-page", response_class=HTMLResponse)
async def predict_page():
    return get_html_response("index.html")


@app.get("/upload-page", response_class=HTMLResponse)
async def upload_page():
    return get_html_response("upload.html")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_package = joblib.load(PROJECT_ROOT / 'models' / 'churn_predictor_pipeline.pkl')
    yield

loaded_pipeline = app.state.model_package['pipeline']
expected_columns = app.state.model_package['features']
num_features = expected_columns['numerical']
cat_features = expected_columns['categorical']

fields = {}
for feature in cat_features:
    fields[feature] = (Optional[str], None) 
for feature in num_features:
    fields[feature] = (Optional[float], None)   

client = create_model('client', **fields)

@app.post("/predict")
async def predict(data: client):
    # data автоматично провалідується. 
    # Перетворюємо в словник, щоб передати в модель
    input_dict = data.model_dump() # або .dict() для старіших версій pydantic
    final_df = pd.DataFrame([input_dict]).reindex(columns=fields.keys())

    try: 
        prediction = loaded_pipeline.predict(final_df)
        probability = loaded_pipeline.predict_proba(final_df)[:, 1]

        return {"status": "success", 
                "churn_prediction": int(prediction[0]),
                'churn_probability': float(probability[0])
        }
    except Exception as e:
        return {"status": "error", "detail": f"Помилка під час прогнозу: {str(e)}"}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Зчитуємо бінарний вміст файлу і декодуємо в текст
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        
        # Завантажуємо в Pandas DataFrame
        df = pd.read_csv(csv_data)
        id_column = 'customerID' if 'customerID' in df.columns else ('customer_id' if 'customer_id' in df.columns else None)
        # Примусово підганяємо типи, як і для поодинокого запиту
        for col in num_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in cat_features:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', None).replace('None', None)
        
        # Залишаємо тільки ті колонки і в тому порядку, який хоче модель
        final_df = df[list(fields.keys())]
        
        # Проганяємо весь датасет через модель!
        predictions = loaded_pipeline.predict(final_df)
        probabilities = loaded_pipeline.predict_proba(final_df)[:,1]
        churn_mask = (predictions ==1 )

        churn_count = int(churn_mask.sum())
        total_rows = len(df)
        churn_customers = []

        if churn_count > 0:
            result_df = pd.DataFrame()
            if id_column:
                result_df['id'] = df.loc[churn_mask, id_column].astype(str)
            else:
                # Якщо унікального ID немає, виводимо індекси рядків для логів
                result_df['id'] = [f"Рядок {i+2}" for i, is_churn in enumerate(churn_mask) if is_churn]
            # Якщо знайшли ID колонку, беремо реальні значення, якщо ні — просто номери рядків
            
            result_df['probability'] = (probabilities[churn_mask] * 100).round(1)
            
            
            # Сортуємо список: спочатку ті, у кого найвищий ризик піти
            churn_customers = result_df.sort_values(by='probability', ascending=False).to_dict('records')


            df_churn = df[churn_mask]
            
            reasons_pool = []
            
            # Перевірка 1: Тип контракту
            if 'Contract' in df_churn.columns:
                m2m_pct = (df_churn['Contract'] == 'Month-to-month').mean() * 100
                if m2m_pct > 50: # Якщо більше половини втікачів на короткому контракті
                    reasons_pool.append({
                        "factor": "Короткострокові контракти (Month-to-month)",
                        "impact": f"{round(m2m_pct, 1)}% клієнтів з групи ризику не мають довгострокових зобов'язань."
                    })
            
            # Перевірка 2: Тип інтернету
            if 'InternetService' in df_churn.columns:
                fiber_pct = (df_churn['InternetService'] == 'Fiber optic').mean() * 100
                if fiber_pct > 40:
                    reasons_pool.append({
                        "factor": "Проблеми з Fiber optic (Оптоволокно)",
                        "impact": f"{round(fiber_pct, 1)}% втікачів користуються оптоволокном. Можливі технічні збої або завищена ціна."
                    })
            
            # Перевірка 3: Техпідтримка
            if 'TechSupport' in df_churn.columns:
                no_support_pct = (df_churn['TechSupport'] == 'No').mean() * 100
                if no_support_pct > 60:
                    reasons_pool.append({
                        "factor": "Відсутність технічної підтримки",
                        "impact": f"{round(no_support_pct, 1)}% людей у зоні ризику не підключили пріоритетну техпідтримку."
                    })
            
            # Перевірка 4: Фінанси (Середній чек)
            if 'MonthlyCharges' in df_churn.columns and 'MonthlyCharges' in df.columns:
                avg_churn_cost = df_churn['MonthlyCharges'].mean()
                avg_total_cost = df['MonthlyCharges'].mean()
                if avg_churn_cost > avg_total_cost:
                    reasons_pool.append({
                        "factor": "Завищена вартість послуг",
                        "impact": f"Середній чек тих, хто йде (${round(avg_churn_cost, 2)}), вищий за середній по компанії (${round(avg_total_cost, 2)})."
                    })

            # Беремо ТОП-3 найважливіших тригери, які вдалося знайти
            top_reasons = reasons_pool[:3]
            if not top_reasons: # Дефолтний варіант, якщо нічого не перевищило ліміти
                top_reasons.append({
                    "factor": "Комплексні фінансові чинники",
                    "impact": "Аналіз вказує на поєднання високих тарифів та низького строку лояльності (tenure)."
                })

        return {
            "status": "success",
            "processed_rows": total_rows,
            "churn_count": churn_count,
            "top_factors": top_reasons, 
            "churn_list": churn_customers
        }
    except Exception as e:
        return {"status": "error", "detail": f"Не вдалося обробити CSV: {str(e)}"}
