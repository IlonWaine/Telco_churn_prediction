import joblib
import pandas as pd
from Config import PROjECT_ROOT
from custon_classes import Missing_data_emulation
from pydantic import create_model
from fastapi import FastAPI , UploadFile, File
from fastapi.responses import HTMLResponse
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO

app = FastAPI()

def get_html_response(file_name: str) -> HTMLResponse:
    with open(PROjECT_ROOT / 'templates' / file_name, "r", encoding="utf-8") as f:
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Дозволяє запити з будь-яких сайтів. Для розробки — супер.
    allow_credentials=True,
    allow_methods=["*"],  # Дозволяє POST, GET тощо.
    allow_headers=["*"],
)

model_package = joblib.load(PROjECT_ROOT / 'models' / 'churn_predictor_pipeline.pkl')

loaded_pipeline = model_package['pipeline']
expected_columns = model_package['features']
num_features = expected_columns['numerical']
cat_features = expected_columns['categorical']

# all_features = cat_features + num_features

# final_model = loaded_pipeline[-1]
# if hasattr(final_model, 'feature_importances_'):
#     importances = final_model.feature_importances_
# df_importance = pd.DataFrame({
#     'Feature': all_features,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# print(df_importance)

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
    new_df = pd.DataFrame([input_dict])
    final_df = pd.DataFrame(columns=fields.keys())
    for col in new_df.columns:
        if col in final_df.columns:
            final_df[col] = new_df[col]
    try: 
        prediction = loaded_pipeline.predict(final_df)
        probability = loaded_pipeline.predict_proba(final_df)[:, 1]
        is_churn = int(prediction[0])
        # churn_res = 'Так' if is_churn == 1 else 'Ні'
        churn_prob = float(probability[0])
        return {"status": "success", 
                "churn_prediction": is_churn,
                'churn_probability': churn_prob
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
        
        churn_count = int((predictions == 1).sum())
        total_rows = len(df)
        churn_customers = []

        if churn_count > 0:
            # Якщо знайшли ID колонку, беремо реальні значення, якщо ні — просто номери рядків
            ids = df[id_column].astype(str).values if id_column else [f"Рядок {i+2}" for i in range(len(df))]
            probs = probabilities
            
            # Збираємо список словників для топ-втікачів
            for i in range(len(df)):
                if churn_mask[i]:
                    churn_customers.append({
                        "id": ids[i],
                        "probability": round(float(probs[i]) * 100, 1) # відсотки, наприклад 84.5%
                    })
            
            # Сортуємо список: спочатку ті, у кого найвищий ризик піти
            churn_customers = sorted(churn_customers, key=lambda x: x['probability'], reverse=True)


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
            "top_factors": top_reasons, # <-- НАДШЕ СТАТИСТИКУ ПРИЧИН
            "churn_list": churn_customers
        }
    except Exception as e:
        return {"status": "error", "detail": f"Не вдалося обробити CSV: {str(e)}"}
