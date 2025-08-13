#Importando Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importando o dataset
df = pd.read_csv('online_shoppers_intention.csv')

#Visualizando algumas informações acerca do dataset
print(df.columns)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

#Como são poucos valores nulos optei por ignorá-los
df.dropna(inplace=True)
print(df.isnull().sum())

#Verificando valores únicos
print(df.nunique())

#Criando uma cópia do df
df_cat = df.copy()

#Listas vazias
continuous = []
categorical = []

#Loop pelas colunas
for col in df.columns[:-1]:
    if df.nunique()[col] >= 30:
        continuous.append(col)
    else:
        categorical.append(col)

#Verificando colunas contínuas e categóricas
print('Contínuas: ', continuous)
print('Categóricas: ', categorical)

#Boxplots
for i, col in enumerate(continuous):
    plt.subplot(3, 3, i + 1);
    df.boxplot(col);
    plt.tight_layout()

#Salvando o Boxplot
plt.savefig('Boxplot1.png')

#Transformação de Log
df[continuous] = np.log1p(1 + df[continuous])

#Boxplots com Log
for i, col in enumerate(continuous):
    plt.subplot(3, 3, i + 1);
    df.boxplot(col);
    plt.tight_layout()

#Salvando o Segundo Boxplot
plt.savefig('Boxplot2.png')

#Heatmap para correlação
plt.figure(figsize=(14, 14))
sns.heatmap(df[continuous].corr(),vmax = 1., square=True);

#Salvando o Heatmap
plt.savefig('Heatmap.png')
plt.show()

#Graficos para variáveis categóricas
plt.title("Venda Efetuada ou Não")
sns.countplot(data = df, x = 'Revenue', hue = 'Revenue');
plt.savefig('Vendas Efetuadas.png')
plt.show()