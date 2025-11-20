# Airbnb Rio — Precificação Inteligente

### 🔗 Deploy da Aplicação

A aplicação já está disponível online em:
**[https://rio-airbnb-predictor-hvmeyhucwspnoozhybvaad.streamlit.app/](https://rio-airbnb-predictor-hvmeyhucwspnoozhybvaad.streamlit.app/)**
(README)

## Visão geral

Este projeto treina e entrega um **modelo de regressão** para prever preços diários de anúncios do Airbnb no Rio de Janeiro. Foi desenvolvido usando Google Colab (treinamento) e uma interface com **Streamlit** para que o cliente possa entrar com os dados do imóvel e obter uma previsão imediata.

> Links de referência usados como base:
>
> * Kaggle (datasets): [https://www.kaggle.com/](https://www.kaggle.com/)
> * scikit-learn (documentação de modelos e utilitários): [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

Coloquei todos os passos necessários para reproduzir o treinamento, preparar os dados, buscar hiperparâmetros, montar um ensemble por *voting* e salvar o artefato final utilizado pela interface.

---

## Objetivos do projeto

1. Unificar 26 datasets (jan–dez de 2018, 2019 e 2020 + total) em uma base coerente.
2. Limpeza e engenharia de features (tratamento de nulos, conversão de categoricas, one-hot encoding, agregações relevantes).
3. Testar e comparar **3 regressors**:

   * HistGradientBoostingRegressor
   * RandomForestRegressor
   * ExtraTreesRegressor
4. Realizar busca de hiperparâmetros (GridSearchCV / RandomizedSearchCV) com validação temporal (TimeSeriesSplit quando fizer sentido).
5. Montar um **VotingRegressor** com os melhores parâmetros e treinar o modelo final.
6. Salvar artefatos importantes (modelo final + scaler + lista de colunas) em `modelo_airbnb.pkl` (formato joblib).
7. Oferecer interface Streamlit que use esse arquivo para prever valores com inputs do usuário.

---

## Estrutura de arquivos (sugerida)

```
project-root/
├─ data/                          # CSVs originais (jan-dez 2018/2019/2020 + total)
├─ notebooks/                      # Colab / Jupyter notebooks (treinamento exploratório)
├─ src/
│  ├─ train.py                     # Script para treinar e salvar modelo
│  ├─ features.py                  # Funções de engenharia de features
│  ├─ preprocess.py                # Pipeline de pré-processamento (ColumnTransformer)
│  └─ utils.py                     # Helpers para leitura e concat
├─ app.py                          # Streamlit app (interface)
├─ modelo_airbnb.pkl               # Artefato salvo (após treinamento)
├─ requirements.txt
└─ README_Airbnb_Rio_Precificacao.md
```

---

## Dados

* Você mencionou que os CSVs estão em `/content/drive/MyDrive/dataset/airbnb` no Colab.
* Recomendo carregar somente as colunas relevantes (remover ids, urls, textos longos).
* Adicionar explicitamente colunas `year` e `month` se ainda não existirem.

**Dica:** ao concatenar arquivos, garanta a mesma ordem e nomes de colunas e use `pd.concat(frames, ignore_index=True)`.

---

## Limpeza e engenharia (passos principais)

1. **Remover colunas inúteis:** ids, urls, descrições longas (a menos que vá extrair texto).
2. **Tratar nulos:**

   * Colunas com muitos nulos (ex.: `review_scores_*`) podem ser removidas ou imputadas com um valor neutro e um flag (ex.: `has_review_scores`).
   * Para numéricos: imputação por mediana.
   * Para categóricas: marcar `unknown` e usar `handle_unknown='ignore'` no OneHotEncoder.
3. **Conversão de tipos:** transformar booleans e flags (`True/False`, `t/f`) para 0/1.
4. **Feature engineering:**

   * `num_amenities` (quantidade de comodidades): já presente em sua interface.
   * Distância até pontos de interesse (opcional): se tiver lat/lon, calcular distância até Copacabana/Ipanema/centro.
   * Interações (ex.: `accommodates * bedrooms`).
5. **Escalonamento:** StandardScaler para features numéricas.
6. **Codificação categórica:** OneHotEncoder com `sparse=False` e `handle_unknown='ignore'`.

---

## Pipeline recomendado (scikit-learn)

Use `ColumnTransformer` para unir transformações numéricas e categóricas.

Exemplo resumido (treinar.py):

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
import joblib

num_cols = ['latitude','longitude','accommodates','bathrooms','bedrooms','beds','extra_people','minimum_nights','num_amenities','host_listings_count','year','month']
cat_cols = ['property_type','room_type','host_is_superhost','instant_bookable']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),
])

# Treinar modelos individualmente com Grid/Random search
models = {
    'hgb': HistGradientBoostingRegressor(random_state=42),
    'rf': RandomForestRegressor(n_jobs=-1, random_state=42),
    'et': ExtraTreesRegressor(n_jobs=-1, random_state=42)
}

# Exemplo de hyperparam grid para RandomizedSearchCV (ajuste conforme necessidade)
param_grid_rf = {
    'n_estimators': [100, 200, 400],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2,5,10]
}

# Use TimeSeriesSplit quando houver dependência temporal
cv = TimeSeriesSplit(n_splits=5)

# Para cada modelo: montar pipeline = preprocessor + model, rodar RandomizedSearchCV
```

Após encontrar os melhores parâmetros para cada regressor, crie pipelines finais com `preprocessor` + `estimator` para cada um. Use as pontuações cross-val para definir pesos do VotingRegressor (ex.: pesos proporcionais ao inverso do RMSE).

---

## Voting e modelo final

1. Monte um `VotingRegressor` com os pipelines finais: `VotingRegressor([('hgb', pipe_hgb), ('rf', pipe_rf), ('et', pipe_et)])`.
2. Treine o `VotingRegressor` no conjunto de treino completo (após validação de hiperparâmetros).
3. Avalie em um holdout (ex.: últimos meses ou última parte da série).
4. Salve os artefatos: scaler (se separado), colunas do modelo (feature names após OneHot) e modelo final.

**Formato sugerido para salvar:**

```python
joblib.dump({'modelo': voting_pipeline, 'scaler': None, 'colunas': feature_names}, 'modelo_airbnb.pkl')
```

No caso do `ColumnTransformer` embutido no pipeline, não é necessário salvar `scaler` separadamente — basta salvar o pipeline.

---

## Sugestões para validação temporal (importante)

* Evite usar `KFold` clássico se os dados têm ordem temporal. Prefira `TimeSeriesSplit` ou validação por blocos (ex.: treinar em 2018–2019, validar em 2020).
* Teste cenários de generalização: treinar em anos anteriores e testar em meses de 2020.

---

## Como rodar (local)

1. Clone o repositório.
2. Criar e ativar ambiente Python (recomendado 3.9+).
3. Instalar dependências:

```
pip install -r requirements.txt
```

`requirements.txt` mínimo:

```
streamlit
pandas
numpy
matplotlib
scikit-learn
joblib
xgboost   # opcional, se for usar
```

4. Rodar app Streamlit localmente (na raiz do projeto):

```
python -m streamlit run app.py
```

---

## Como rodar (Colab)

* Abra o notebook de treinamento (você já compartilhou o link do Colab). Monte o Drive, rode `train.py` ou execute as células do notebook. Salve `modelo_airbnb.pkl` no Drive e baixe para a pasta do app antes de subir.

---

## 📌 Treinamento e Origem dos Dados

O modelo foi treinado diretamente no Google Colab:

* Colab: [https://colab.research.google.com/drive/16jWT35SYl6NPKarvkeMHF0TnzI2877oG?usp=sharing](https://colab.research.google.com/drive/16jWT35SYl6NPKarvkeMHF0TnzI2877oG?usp=sharing)

Datasets utilizados estão disponíveis tanto no Google Drive quanto no Kaggle:

* Drive: [https://drive.google.com/drive/folders/1HGr7xnseMiajB-IB9xTYEhkktULqOPdt?usp=drive_link](https://drive.google.com/drive/folders/1HGr7xnseMiajB-IB9xTYEhkktULqOPdt?usp=drive_link)
* Kaggle: [https://www.kaggle.com/code/eduardoferreirasilva/airbnb-rio-ferramenta-de-predi-o-de-pre-os/input?select=agosto2019.csv](https://www.kaggle.com/code/eduardoferreirasilva/airbnb-rio-ferramenta-de-predi-o-de-pre-os/input?select=agosto2019.csv)
