import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==================================================
# CONFIGURAÇÃO DA PÁGINA
# ==================================================
st.set_page_config(
    page_title="Airbnb Rio - Precificação IA",
    page_icon="🏖️",
    layout="wide"
)

# Estilo CSS para deixar o botão e títulos mais bonitos
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; color: #FF5A5F; }
    .stButton>button { background-color: #FF5A5F; color: white; width: 100%; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# 1. CARREGAMENTO DO MODELO (COM CACHE)
# ==================================================
@st.cache_resource
def carregar_inteligencia():
    try:
        # Tenta carregar o dicionário salvo
        dados = joblib.load("modelo_airbnb.pkl")
        return dados
    except FileNotFoundError:
        return None

pacote = carregar_inteligencia()

if pacote is None:
    st.error("❌ Erro Crítico: O arquivo 'modelo_airbnb.pkl' não foi encontrado.")
    st.info("Certifique-se de ter rodado o treinamento no Colab e baixado o arquivo para a mesma pasta deste script.")
    st.stop()

model = pacote['modelo']
scaler = pacote['scaler']
colunas_modelo = pacote['colunas']

# ==================================================
# 2. BARRA LATERAL (INPUTS)
# ==================================================
st.sidebar.header("🏠 Configurar Imóvel")

# --- A. Localização ---
st.sidebar.subheader("1. Localização")
bairros_rio = {
    "Copacabana (Posto 4)": (-22.9711, -43.1889),
    "Copacabana (Posto 6)": (-22.9813, -43.1920),
    "Ipanema (Posto 9)": (-22.9847, -43.2044),
    "Leblon": (-22.9866, -43.2233),
    "Barra (Jardim Oceânico)": (-23.0118, -43.3049),
    "Barra (Centro)": (-23.0005, -43.3564),
    "Recreio": (-23.0289, -43.4664),
    "Botafogo": (-22.9514, -43.1809),
    "Flamengo": (-22.9338, -43.1762),
    "Centro / Lapa": (-22.9134, -43.1809),
    "Santa Teresa": (-22.9226, -43.1866),
    "Tijuca": (-22.9238, -43.2334),
}

local_escolhido = st.sidebar.selectbox("Região de Referência", list(bairros_rio.keys()))
lat_padrao, lon_padrao = bairros_rio[local_escolhido]

# Permite ajuste fino
lat = st.sidebar.number_input("Latitude", value=lat_padrao, format="%.5f")
lon = st.sidebar.number_input("Longitude", value=lon_padrao, format="%.5f")

# --- B. Detalhes Físicos ---
st.sidebar.subheader("2. Detalhes Físicos")
tipo_imovel = st.sidebar.selectbox("Tipo", ["Apartment", "House", "Condominium", "Loft", "Outros"])
tipo_quarto = st.sidebar.selectbox("Privacidade", ["Entire home/apt", "Private room", "Shared room"])

accommodates = st.sidebar.slider("Pessoas", 1, 15, 4)
quartos = st.sidebar.number_input("Quartos", 0, 10, 1)
banheiros = st.sidebar.number_input("Banheiros", 1, 5, 1)
camas = st.sidebar.number_input("Camas", 1, 10, 2)

# --- C. Extras e Regras ---
st.sidebar.subheader("3. Extras e Regras")
amenities = st.sidebar.slider("Qtd. Comodidades (Wifi, TV, AC...)", 0, 60, 20)
extra_people = st.sidebar.number_input("Taxa hóspede extra (R$)", 0.0, 300.0, 40.0)
min_nights = st.sidebar.number_input("Mínimo Noites", 1, 30, 3)

is_superhost = st.sidebar.checkbox("É Superhost?")
instant_book = st.sidebar.checkbox("Reserva Instantânea?")

# ==================================================
# 3. CORPO PRINCIPAL
# ==================================================
st.title("🏖️ Precificação Inteligente Airbnb Rio")
st.markdown(f"**Análise para:** {local_escolhido} ({lat}, {lon})")

col_top1, col_top2 = st.columns(2)
with col_top1:
    st.info("Defina os parâmetros na barra lateral e clique em calcular.")
with col_top2:
    # Inputs de Projeção (Máquina do Tempo)
    ano_alvo = st.number_input("📅 Projetar valorização para o ano:", 2024, 2035, 2025)
    taxa_inflacao = st.slider("📈 Taxa Anual Esperada (%)", 0, 20, 6)

st.divider()

if st.button("CALCULAR PREÇO SUGERIDO 🚀"):
    
    # --- PASSO 1: Construir o DataFrame de Entrada ---
    # Criamos uma linha zerada com TODAS as colunas que o modelo conhece
    entrada = pd.DataFrame(0, index=[0], columns=colunas_modelo)
    
    # --- PASSO 2: Preencher Numéricos (COM BLINDAGEM) ---
    # Só preenchemos se a coluna existir no modelo treinado (Evita erro "unseen feature")
    
    if 'latitude' in entrada.columns: entrada['latitude'] = lat
    if 'longitude' in entrada.columns: entrada['longitude'] = lon
    if 'accommodates' in entrada.columns: entrada['accommodates'] = accommodates
    if 'bathrooms' in entrada.columns: entrada['bathrooms'] = banheiros
    if 'bedrooms' in entrada.columns: entrada['bedrooms'] = quartos
    if 'beds' in entrada.columns: entrada['beds'] = camas
    if 'extra_people' in entrada.columns: entrada['extra_people'] = extra_people
    if 'minimum_nights' in entrada.columns: entrada['minimum_nights'] = min_nights
    if 'num_amenities' in entrada.columns: entrada['num_amenities'] = amenities
    if 'host_listings_count' in entrada.columns: entrada['host_listings_count'] = 3
    
    # Datas: Se o modelo não tiver 'year', ele ignora e não dá erro
    if 'year' in entrada.columns: entrada['year'] = datetime.now().year
    if 'month' in entrada.columns: entrada['month'] = datetime.now().month
    
    # --- PASSO 3: Preencher Categorias (One-Hot Encoding) ---
    def marcar_coluna(prefixo, valor):
        nome_coluna = f"{prefixo}_{valor}"
        # Proteção: Só marca se a coluna existir
        if nome_coluna in entrada.columns:
            entrada[nome_coluna] = 1
    
    marcar_coluna('property_type', tipo_imovel)
    marcar_coluna('room_type', tipo_quarto)
    
    # Proteção para Booleanos (tenta as duas formas comuns)
    if is_superhost:
        if 'host_is_superhost_t' in entrada.columns: entrada['host_is_superhost_t'] = 1
        elif 'host_is_superhost_True' in entrada.columns: entrada['host_is_superhost_True'] = 1
        
    if instant_book:
        if 'instant_bookable_t' in entrada.columns: entrada['instant_bookable_t'] = 1
        elif 'instant_bookable_True' in entrada.columns: entrada['instant_bookable_True'] = 1

    # --- PASSO 4: Previsão Base ---
    try:
        # Escalar os dados
        entrada_scaled = scaler.transform(entrada)
        preco_hoje = model.predict(entrada_scaled)[0]
    except Exception as e:
        st.error(f"Erro ao realizar a predição: {e}")
        st.stop()
    
    # --- PASSO 5: Projeção Futura (Matemática Financeira) ---
    anos_dif = ano_alvo - datetime.now().year
    if anos_dif < 0: anos_dif = 0
    preco_futuro = preco_hoje * ((1 + taxa_inflacao/100) ** anos_dif)
    
    # --- RESULTADOS VISUAIS ---
    
    # 1. Métricas Principais
    c1, c2, c3 = st.columns(3)
    c1.metric("Preço Diária (Hoje)", f"R$ {preco_hoje:.2f}")
    c2.metric(f"Preço em {ano_alvo}", f"R$ {preco_futuro:.2f}", delta=f"+{anos_dif} anos")
    c3.metric("Potencial Mensal (50% Ocupação)", f"R$ {preco_hoje * 15:.2f}")
    
    # 2. Gráfico de Sazonalidade
    st.subheader("📊 Sazonalidade (Variação durante o ano)")
    
    # Só faz sentido mostrar sazonalidade se o modelo conhecer a coluna 'month'
    if 'month' in entrada.columns:
        df_sazonal = pd.concat([entrada]*12, ignore_index=True)
        df_sazonal['month'] = range(1, 13) # Jan a Dez
        
        sazonal_scaled = scaler.transform(df_sazonal)
        precos_ano = model.predict(sazonal_scaled)
        
        meses_nome = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(meses_nome, precos_ano, marker='o', linestyle='-', color='#FF5A5F', linewidth=2)
        ax.fill_between(meses_nome, precos_ano, color='#FF5A5F', alpha=0.1)
        ax.set_ylabel("Preço (R$)")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
    else:
        st.warning("O modelo atual não foi treinado com dados mensais, então não podemos prever a sazonalidade exata.")
    
    # 3. Gráfico de Projeção Anual
    st.subheader(f"📈 Curva de Valorização até {ano_alvo}")
    dados_futuro = []
    anos_lista = list(range(datetime.now().year, ano_alvo + 1))
    
    for i, ano in enumerate(anos_lista):
        valor = preco_hoje * ((1 + taxa_inflacao/100) ** i)
        dados_futuro.append(valor)
        
    df_chart_futuro = pd.DataFrame({"Ano": anos_lista, "Valor Estimado": dados_futuro})
    st.line_chart(df_chart_futuro.set_index("Ano"))

else:
    st.write("👈 Preencha os dados ao lado para gerar a previsão.")