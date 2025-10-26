import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Configuração da página
st.set_page_config(layout="wide", page_title="Análise de Satisfação Aérea")

# --- 1. Carregamento e Cache dos Dados ---

@st.cache_data
def load_data(train_path, test_path):
    """Carrega e limpa os dados de treino e teste."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Remove colunas desnecessárias que o seu código original removeu (Unnamed: 0, id)
    df_train = df_train.drop(columns=[col for col in df_train.columns if 'Unnamed' in col or col == 'id'], errors='ignore')
    df_test = df_test.drop(columns=[col for col in df_test.columns if 'Unnamed' in col or col == 'id'], errors='ignore')

    # Tratamento de NaNs (substituindo pela mediana no caso de 'Arrival Delay in Minutes')
    # O RF até lidaria com NaNs, mas o get_dummies não. É melhor tratar.
    for df in [df_train, df_test]:
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())

    # Remove qualquer outra linha com NaN (para garantir) - nao tem outra entao ok 
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)
    
    return df_train, df_test

# --- 2. Função de Preprocessamento ---

def preprocess(df):
    """Codifica o dataframe para o modelo."""
    df_processed = df.copy()
    
    # Mapeia a satisfação
    if 'satisfaction' in df_processed.columns:
        df_processed['satisfaction'] = df_processed['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
    
    # Cria dummies para todas as colunas categóricas
    df_processed = pd.get_dummies(df_processed, drop_first=True)
    
    return df_processed

# --- 3. Treinamento do Modelo (com Cache) ---

@st.cache_resource
def train_model(X_train, y_train):
    """Treina o modelo Random Forest."""
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


@st.cache_data
def analyze_nas(data_path):
    """
    Carrega os dados brutos e retorna uma tabela (DataFrame) 
    com a contagem e porcentagem de valores NAs por coluna.
    """
    # Carrega os dados brutos, ANTES de qualquer tratamento
    df_raw = pd.read_csv(data_path)
    
    # Calcula a contagem de NAs
    na_counts = df_raw.isna().sum()
    
    # Calcula a porcentagem de NAs
    na_percentage = (na_counts / len(df_raw)) * 100
    
    # Cria a tabela de resumo
    na_summary_df = pd.DataFrame({
        'Coluna': na_counts.index,
        'Qtd. de NAs': na_counts.values,
        '% de NAs': na_percentage.values
    })
    
    # Filtra para mostrar apenas colunas que REALMENTE têm NAs
    na_summary_df = na_summary_df[na_summary_df['Qtd. de NAs'] > 0]
    
    # Ordena para mostrar as colunas com mais NAs primeiro
    na_summary_df = na_summary_df.sort_values(by='% de NAs', ascending=False)
    
    # Formata a porcentagem para ficar mais bonita
    na_summary_df['% de NAs'] = na_summary_df['% de NAs'].map('{:.2f}%'.format)
    
    return na_summary_df

# --- 4. Carregamento e Preparação Principal ---
df_train, df_test = load_data('train.csv', 'test.csv')

#analisa os NAs do treino e teste 
na_summary_table_train = analyze_nas('train.csv')
na_summary_table_test = analyze_nas('test.csv')

# Calcula o sumário estatístico (min, max, média, etc.) do df_train
# .round(2) arredonda a média para 2 casas decimais
data_summary = df_train.describe().round(2)

# Processa os dados
train_encoded = preprocess(df_train)
test_encoded = preprocess(df_test)

# Alinha colunas: Garante que teste e treino tenham as mesmas colunas dummies
train_cols = set(train_encoded.columns)
test_cols = set(test_encoded.columns)

missing_in_test = list(train_cols - test_cols)
if 'satisfaction' in missing_in_test:
    missing_in_test.remove('satisfaction') # Remove o target se ele estiver faltando (o que é esperado)
for col in missing_in_test:
    test_encoded[col] = 0 # Adiciona colunas faltantes no teste com 0

missing_in_train = list(test_cols - train_cols)
if 'satisfaction' in missing_in_train:
    missing_in_train.remove('satisfaction')
for col in missing_in_train:
    train_encoded[col] = 0 # Adiciona colunas faltantes no treino com 0
    
# Garante a ordem das colunas
test_encoded = test_encoded[train_encoded.drop('satisfaction', axis=1).columns.tolist() + ['satisfaction']]


# Separa feature e target
X_train = train_encoded.drop("satisfaction", axis=1)
y_train = train_encoded["satisfaction"]

X_test = test_encoded.drop("satisfaction", axis=1)
y_test = test_encoded["satisfaction"]

# Treina o modelo
rf_model = train_model(X_train, y_train)

# Gera predições e métricas
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1] # Probabilidade de "satisfeito"

# Calcula Feature Importances
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top15 = importances.sort_values(ascending=False).head(15)


# --- 5. Layout do Aplicativo Streamlit ---

st.title("✈️ Dashboard de Satisfação de Passageiros Aéreos")
st.markdown("Este dashboard interativo analisa dados de satisfação de passageiros e permite prever se um cliente ficará satisfeito com base em suas características e avaliações de serviço.")

# --- Barra Lateral para o Simulador ---
st.sidebar.header("Simulador de Satisfação")
st.sidebar.markdown("Insira os dados do passageiro para prever a satisfação:")

# Criar inputs na sidebar. Usamos os valores únicos do df_train original
input_data = {}

# Colunas categóricas (seleção)
cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for col in cat_cols:
    options = df_train[col].unique()
    input_data[col] = st.sidebar.selectbox(f"{col}", options=options)

# Colunas numéricas (sliders ou number_input)
num_cols = {
    'Age': (10, 85, 40),
    'Flight Distance': (0, 5000, 1500),
    'Inflight wifi service': (0, 5, 3),
    'Departure/Arrival time convenient': (0, 5, 3),
    'Ease of Online booking': (0, 5, 3),
    'Gate location': (0, 5, 3),
    'Food and drink': (0, 5, 3),
    'Online boarding': (0, 5, 3),
    'Seat comfort': (0, 5, 3),
    'Inflight entertainment': (0, 5, 3),
    'On-board service': (0, 5, 3),
    'Leg room service': (0, 5, 3),
    'Baggage handling': (0, 5, 3),
    'Checkin service': (0, 5, 3),
    'Inflight service': (0, 5, 3),
    'Cleanliness': (0, 5, 3),
    'Departure Delay in Minutes': (0, 300, 0),
    'Arrival Delay in Minutes': (0, 300, 0)
}

st.sidebar.subheader("Avaliações (0-5) e Outros Dados")
for col, (min_val, max_val, default_val) in num_cols.items():
    if col in ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']:
        input_data[col] = st.sidebar.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default_val)
    else:
        input_data[col] = st.sidebar.slider(f"{col}", min_value=min_val, max_value=max_val, value=default_val)

# Botão para prever
if st.sidebar.button("Prever Satisfação"):
    # 1. Criar DataFrame com os inputs
    input_df = pd.DataFrame([input_data])
    
    # 2. Processar o DataFrame (criar dummies)
    input_processed = pd.get_dummies(input_df, drop_first=True)
    
    # 3. Alinhar colunas com X_train
    # Reindexa para garantir que o input tenha exatamente as mesmas colunas que o modelo espera
    input_final = input_processed.reindex(columns=X_train.columns, fill_value=0)
    
    # 4. Fazer a predição
    prediction = rf_model.predict(input_final)[0]
    prediction_proba = rf_model.predict_proba(input_final)[0][1] # Prob de estar satisfeito
    
    # 5. Mostrar o resultado
    st.sidebar.subheader("Resultado da Previsão:")
    if prediction == 1:
        st.sidebar.success(f"**Cliente Satisfeito** (Probabilidade: {prediction_proba:.1%})")
    else:
        st.sidebar.error(f"**Cliente Neutro ou Insatisfeito** (Probabilidade de satisfação: {prediction_proba:.1%})")


# --- Abas Principais ---
tab0, tab1, tab2, tab3 = st.tabs(["📃 Informações das Variáveis", "📊 Análise Exploratória", "📈 Performance do Modelo", "💡 Fatores de Influência"])

with tab0: 
    st.header("Glossário das Variáveis do Dataset")
    st.markdown("Aqui está a descrição de todas as colunas usadas na análise, "
                "incluindo seus valores mínimos, máximos e médios calculados a partir dos dados de treino.")

    # --- Grupo 1: Infos do Passageiro ---
    with st.expander("👤 Informações do Passageiro"):
        st.markdown(f"""
            ### Gender
            **Descrição:** Gênero do passageiro.
            - **Valores:** `Female` (Feminino), `Male` (Masculino)

            ### Customer Type
            **Descrição:** Tipo de cliente.
            - **Valores:** `Loyal customer` (Cliente Fiel), `disloyal customer` (Cliente Ocasional)

            ### Age
            **Descrição:** Idade real do passageiro.
            - **Mínimo:** `{data_summary.loc['min', 'Age']}`
            - **Média:** `{data_summary.loc['mean', 'Age']}`
            - **Máximo:** `{data_summary.loc['max', 'Age']}`

            ### Type of Travel
            **Descrição:** Propósito do voo.
            - **Valores:** `Personal Travel` (Viagem Pessoal), `Business Travel` (Viagem de Negócios)

            ### Class
            **Descrição:** Classe da viagem no avião.
            - **Valores:** `Business`, `Eco`, `Eco Plus`
        """)
    
    # --- Grupo 2: Avaliações de Serviço ---
    with st.expander("⭐ Avaliações de Serviço (Notas 0-5)"):
        st.info("Para estas colunas: **0** significa 'Não Aplicável' e **1-5** são os níveis de satisfação (de Pior para Melhor).")
        st.markdown(f"""
            ### Inflight wifi service
            **Descrição:** Satisfação com o **Wi-Fi** a bordo.
            - **Mínimo:** `{data_summary.loc['min', 'Inflight wifi service']}`
            - **Média:** `{data_summary.loc['mean', 'Inflight wifi service']}`
            - **Máximo:** `{data_summary.loc['max', 'Inflight wifi service']}`

            ### Departure/Arrival time convenient
            **Descrição:** Satisfação com a **conveniência dos horários** de partida/chegada.
            - **Mínimo:** `{data_summary.loc['min', 'Departure/Arrival time convenient']}`
            - **Média:** `{data_summary.loc['mean', 'Departure/Arrival time convenient']}`
            - **Máximo:** `{data_summary.loc['max', 'Departure/Arrival time convenient']}`
            
            ### Ease of Online booking
            **Descrição:** Satisfação com a **facilidade da reserva online**.
            - **Mínimo:** `{data_summary.loc['min', 'Ease of Online booking']}`
            - **Média:** `{data_summary.loc['mean', 'Ease of Online booking']}`
            - **Máximo:** `{data_summary.loc['max', 'Ease of Online booking']}`
            
            ### Gate location
            **Descrição:** Satisfação com a **localização do portão** de embarque.
            - **Mínimo:** `{data_summary.loc['min', 'Gate location']}`
            - **Média:** `{data_summary.loc['mean', 'Gate location']}`
            - **Máximo:** `{data_summary.loc['max', 'Gate location']}`
            
            ### Food and drink
            **Descrição:** Satisfação com a **comida e bebida**.
            - **Mínimo:** `{data_summary.loc['min', 'Food and drink']}`
            - **Média:** `{data_summary.loc['mean', 'Food and drink']}`
            - **Máximo:** `{data_summary.loc['max', 'Food and drink']}`
            
            ### Online boarding
            **Descrição:** Satisfação com o **embarque online** (via app/web).
            - **Mínimo:** `{data_summary.loc['min', 'Online boarding']}`
            - **Média:** `{data_summary.loc['mean', 'Online boarding']}`
            - **Máximo:** `{data_summary.loc['max', 'Online boarding']}`
            
            ### Seat comfort
            **Descrição:** Satisfação com o **conforto do assento**.
            - **Mínimo:** `{data_summary.loc['min', 'Seat comfort']}`
            - **Média:** `{data_summary.loc['mean', 'Seat comfort']}`
            - **Máximo:** `{data_summary.loc['max', 'Seat comfort']}`
            
            ### Inflight entertainment
            **Descrição:** Satisfação com o **entretenimento a bordo**.
            - **Mínimo:** `{data_summary.loc['min', 'Inflight entertainment']}`
            - **Média:** `{data_summary.loc['mean', 'Inflight entertainment']}`
            - **Máximo:** `{data_summary.loc['max', 'Inflight entertainment']}`
            
            ### On-board service
            **Descrição:** Satisfação com o **serviço de bordo** (atendimento da tripulação).
            - **Mínimo:** `{data_summary.loc['min', 'On-board service']}`
            - **Média:** `{data_summary.loc['mean', 'On-board service']}`
            - **Máximo:** `{data_summary.loc['max', 'On-board service']}`
            
            ### Leg room service
            **Descrição:** Satisfação com o **espaço para as pernas**.
            - **Mínimo:** `{data_summary.loc['min', 'Leg room service']}`
            - **Média:** `{data_summary.loc['mean', 'Leg room service']}`
            - **Máximo:** `{data_summary.loc['max', 'Leg room service']}`
            
            ### Baggage handling
            **Descrição:** Satisfação com o **manuseio da bagagem**.
            - **Mínimo:** `{data_summary.loc['min', 'Baggage handling']}`
            - **Média:** `{data_summary.loc['mean', 'Baggage handling']}`
            - **Máximo:** `{data_summary.loc['max', 'Baggage handling']}`
            
            ### Checkin service
            **Descrição:** Satisfação com o **serviço de check-in**.
            - **Mínimo:** `{data_summary.loc['min', 'Checkin service']}`
            - **Média:** `{data_summary.loc['mean', 'Checkin service']}`
            - **Máximo:** `{data_summary.loc['max', 'Checkin service']}`
            
            ### Inflight service
            **Descrição:** Satisfação com o **serviço durante o voo** (ex: entrega de refeições).
            - **Mínimo:** `{data_summary.loc['min', 'Inflight service']}`
            - **Média:** `{data_summary.loc['mean', 'Inflight service']}`
            - **Máximo:** `{data_summary.loc['max', 'Inflight service']}`
            
            ### Cleanliness
            **Descrição:** Satisfação com a **limpeza** da aeronave.
            - **Mínimo:** `{data_summary.loc['min', 'Cleanliness']}`
            - **Média:** `{data_summary.loc['mean', 'Cleanliness']}`
            - **Máximo:** `{data_summary.loc['max', 'Cleanliness']}`
        """)
    
    # --- Grupo 3: Métricas de Voo ---
    with st.expander("✈️ Métricas do Voo e Atrasos"):
        st.markdown(f"""
            ### Flight Distance
            **Descrição:** A distância do voo.
            - **Mínimo:** `{data_summary.loc['min', 'Flight Distance']}`
            - **Média:** `{data_summary.loc['mean', 'Flight Distance']}`
            - **Máximo:** `{data_summary.loc['max', 'Flight Distance']}`

            ### Departure Delay in Minutes
            **Descrição:** Minutos de atraso na partida.
            - **Mínimo:** `{data_summary.loc['min', 'Departure Delay in Minutes']}`
            - **Média:** `{data_summary.loc['mean', 'Departure Delay in Minutes']}`
            - **Máximo:** `{data_summary.loc['max', 'Departure Delay in Minutes']}`
            
            ### Arrival Delay in Minutes
            **Descrição:** Minutos de atraso na chegada.
            - **Mínimo:** `{data_summary.loc['min', 'Arrival Delay in Minutes']}`
            - **Média:** `{data_summary.loc['mean', 'Arrival Delay in Minutes']}`
            - **Máximo:** `{data_summary.loc['max', 'Arrival Delay in Minutes']}`
        """)

    # --- Grupo 4: Variável Alvo ---
    with st.expander("🎯 Variável Alvo (Target)"):
        st.markdown("""
            ### Satisfaction
            **Descrição:** Nível de satisfação geral com a companhia aérea (esta é a variável que o modelo tenta prever).
            - **Valores:** `satisfied` (Satisfeito), `neutral or dissatisfied` (Neutro ou Insatisfeito)
        """)
        
    st.warning(
    """
    - Os dados apresentados das métricas são do dataset de treino.
    - Para variáveis categóricas, realizamos o one-hot encoding.
    """
)

with tab1:
    st.header("Análise Exploratória dos Dados")
    st.markdown("Você pode encontrar a base original nesse [link](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)")    
    st.subheader("Análise de Valores Faltantes (NAs)")
    st.markdown("Valores faltantes encontrados nos dados brutos (antes do tratamento):")
    
    st.markdown("Treino")
    if na_summary_table_train.empty:
        st.success("🎉 Ótimo! Nenhuma coluna com valores faltantes foi encontrada.")
    else:
        st.dataframe(na_summary_table_train, use_container_width=True)

    st.markdown("Teste")
    if na_summary_table_test.empty:
        st.success("🎉 Ótimo! Nenhuma coluna com valores faltantes foi encontrada.")
    else:
        st.dataframe(na_summary_table_test, use_container_width=True)
        
    st.info("Substituimos pela mediana para tratar os NAs")
    
    st.warning("A partir daqui só realizamos com os dados de treino")

    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição da Satisfação")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df_train, x="satisfaction", palette="viridis", ax=ax1)
        ax1.set_title("Distribuição de Satisfação dos Passageiros")
        st.pyplot(fig1)

    with col2:
        st.subheader("Taxa de Satisfação por Classe")
        fig2, ax2 = plt.subplots()
        train_plot = df_train.copy()
        train_plot['satisfaction'] = train_plot['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
        sns.barplot(data=train_plot, x="Class", y="satisfaction", estimator=np.mean, palette="viridis", ax=ax2)
        ax2.set_title("Taxa de Satisfação por Classe")
        ax2.set_ylabel("Proporção de Satisfeitos")
        st.pyplot(fig2)

    st.subheader("Mapa de Correlação (Após Encoding)")
    st.markdown("Correlações entre as variáveis numéricas e *dummies*. Ajuda a ver relações lineares.")
    train_corr = pd.get_dummies(df_train, drop_first=True)
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    # Selecionar apenas colunas numéricas para o heatmap
    numeric_cols = train_corr.select_dtypes(include=np.number).columns
    sns.heatmap(train_corr[numeric_cols].corr(), cmap='coolwarm', center=0, ax=ax3)
    ax3.set_title("Mapa de Correlação")
    st.pyplot(fig3)


with tab2:
    st.header("Performance do Modelo: Random Forest")
    st.markdown(f"O modelo foi treinado com `{X_train.shape[0]}` amostras e testado com `{X_test.shape[0]}` amostras.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Métricas Principais")
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        st.metric(label="**Acurácia**", value=f"{acc:.2%}")
        st.metric(label="**AUC (ROC)**", value=f"{auc:.3f}")
        
        st.subheader("Relatório de Classificação")
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        st.dataframe(report_df.round(3))
        
    with col2:
        st.subheader("Matriz de Confusão")
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                    xticklabels=['Neutro/Insatisfeito', 'Satisfeito'],
                    yticklabels=['Neutro/Insatisfeito', 'Satisfeito'])
        ax4.set_title("Matriz de Confusão")
        ax4.set_xlabel("Previsto")
        ax4.set_ylabel("Real")
        st.pyplot(fig4)

with tab3:
    st.header("💡 Principais Fatores de Influência")
    st.markdown("Quais fatores o modelo (Random Forest) considerou mais importantes para prever a satisfação do cliente?")
    
    fig5, ax5 = plt.subplots(figsize=(10, 7))
    sns.barplot(x=top15.values, y=top15.index, palette='viridis', ax=ax5)
    ax5.set_title("Top 15 Fatores que Mais Influenciam a Satisfação")
    ax5.set_xlabel("Importância no Modelo")
    ax5.set_ylabel("Fator")
    st.pyplot(fig5)
    
    st.subheader("Top 5 Fatores:")
    st.info(f"""
    1. **{top15.index[0]}**
    2. **{top15.index[1]}**
    3. **{top15.index[2]}**
    4. **{top15.index[3]}**
    5. **{top15.index[4]}**
    """)
    st.markdown("""
    **Interpretação:** O modelo indica que fatores como `Online boarding`, `Inflight wifi service`, e o `Tipo de Viagem` (se é a negócios ou pessoal) são os preditores mais fortes da satisfação. A classe (`Class_Eco`, `Class_Eco Plus`) também tem um peso significativo.
    """)







    