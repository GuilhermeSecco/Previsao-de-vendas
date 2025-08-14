# 💰 Previsão de Compras Online com SVM

#### Este projeto utiliza Support Vector Machines (SVM) para prever se uma compra online será ou não efetuada, a partir de dados de sessões de usuários em um e-commerce. Foram explorados diferentes kernels e técnicas de pré-processamento para otimizar a performance dos modelos.

## 📂 Dataset

#### O conjunto de dados utilizado é o Online Shoppers Purchasing Intention Dataset, que contém informações sobre comportamento de navegação, atributos temporais, dados de visitante e se a compra foi finalizada (Revenue).

Fonte: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset)

# 📊 Etapas do Projeto
## 1. Análise Exploratória de Dados (EDA)

#### Inspeção inicial de colunas, tipos e valores nulos.

#### Identificação de variáveis contínuas e categóricas.

#### Visualizações:

    Boxplots (antes e depois da transformação logarítmica).

    Heatmap de correlação para variáveis contínuas.
  
    Distribuição de vendas (Revenue) e tipos de visitantes (VisitorType).

## 2. Pré-Processamento

#### Tratamento de valores nulos (remoção direta).

#### Codificação de variáveis categóricas (LabelEncoder para Month e VisitorType).

#### Balanceamento da variável alvo usando SMOTE para evitar viés do modelo.

#### Padronização dos dados com StandardScaler (para modelos específicos).

## 3. Modelagem

### Foram treinados quatro modelos de SVM:

  |Versão|Kernel|Pré-processamento|Observações|
  |:---:|:---:|:---:|:---:|
  |V1|Linear|Sem padronização|Modelo simples base|
  |V2|Linear|Dados padronizados|Melhoria na convergência
  |V3|RBF|Padronização + GridSearchCV|Busca de hiperparâmetros ideais (C e gamma)|
  |V4|Polinomial|Padronização + GridSearchCV|Ajuste de gamma, degree e coef0|

## 4. Avaliação de Desempenho

### Métricas utilizadas:

    Precision

    Recall

    F1-score

    Accuracy

    AUC

    Matriz de confusão

### Resultados resumidos:

  ||Modelo 1|Modelo 2|Modelo 3|Modelo 4|
  |:---|---:|---:|---:|---:|
  |Modelo|            SVM|         SVM|         SVM|         SVM|
  |Kernel|         Linear|      Linear|         RBF|        Poly|
  |Precision|    0.794702|    0.796594|    0.852412|    0.925576|
  |Recall|       0.889831|     0.88975|     0.88623|     0.88377|
  |F1|            0.83958|    0.840599|    0.868992|     0.90419|
  |Accuracy|     0.845797|    0.846597|    0.869496|      0.9004|
  |AUC|          0.849545|    0.850189|    0.869875|    0.901529|

## 5. Visualizações Geradas

### O projeto salva automaticamente alguns gráficos durante a execução:

    Boxplot1.png – Boxplots das variáveis contínuas (original).

    Boxplot2.png – Boxplots das variáveis contínuas após transformação log.

    Heatmap.png – Mapa de correlação.

    Vendas Efetuadas.png – Distribuição de compras finalizadas ou não.

    Tipos de Visitantes.png – Distribuição de tipos de visitantes.

    SMOTE.png – Balanceamento da variável alvo após oversampling.

    comparacao_modelos_svm.png - Visualização dos resultados obtidos

# 🛠 Tecnologias Utilizadas

    Python 3

    Pandas, NumPy – Manipulação e análise de dados.

    Matplotlib, Seaborn – Visualizações.

    Scikit-learn – Modelagem SVM, métricas, pré-processamento.

    Imbalanced-learn (SMOTE) – Balanceamento da variável alvo.

# 🚀 Como Executar

### 1. Clone este repositório:
    git clone https://github.com/GuilhermeSecco/Previsao-de-vendas.git
    cd Previsao-de-vendas

### 2. Instale as dependências:
    pip install -r requirements.txt

### 3. Adicione o arquivo online_shoppers_intention.csv na pasta do projeto.

## 4. Execute o script principal:
    python Previsor-de-vendas.py

# 📌 Conclusão

#### O uso de SVM demonstrou boa capacidade de classificação para prever compras online, principalmente após o balanceamento de dados e ajuste de hiperparâmetros. O kernel RBF obteve melhor desempenho geral no conjunto de teste, por fim o kernel linear padronizado também apresentou resultados consistentes com menor custo computacional.
