#Importando Bibliotecas
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn import svm
import sklearn
import matplotlib
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

#Análise Exploratória

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

#Tipos de Visitantes
plt.xlabel("Tipos de Visitantes")
sns.countplot(data = df, x = 'VisitorType', hue = 'VisitorType');
plt.savefig('Tipos de Visitantes.png')
plt.show()

#Pré-Processamento dos Dados

#Encoder
lb = LabelEncoder()

#Aplicando o encoder nas variáveis com string
df['Month'] = lb.fit_transform(df['Month'])
df['VisitorType'] = lb.fit_transform(df['VisitorType'])

#Removendo valores nulos caso tenham sido gerados
df.dropna(inplace=True)

#Verificando os registros
print(df.head(20))
print(df.shape)

#Verificando o Balanceamento da variável de vendas
target_count = df['Revenue'].value_counts()
print(target_count)

#Importando a função de Oversampling
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE

#Seed para reprodução do mesmo resultado
seed = 9

#Separando X e y
X = df.iloc[:, 0:17]
y = df.iloc[:, 17]

#Criando balanceador SMOTE
sm = SMOTE(random_state=seed)

#Aplicando o Balanceador
X_res, y_res = sm.fit_resample(X, y)

#Verificando a variável alvo
target_count_smote = y_res.value_counts()
print(target_count_smote)
plt.title('SMOTE')
sns.countplot(data = target_count_smote, palette='Set1');
plt.savefig('SMOTE.png')
plt.show()

#Redefinindo X e y
X = X_res
y = y_res

#Separando entre Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

#SVM Model
#Modelo de Kernel Linear
model_v1 = svm.SVC(kernel = 'linear')
start = time.time()
model_v1.fit(X_train, y_train)
end = time.time()
print("Duração do treino do Modelo", end - start)

#Previsões
predictions_v1 = model_v1.predict(X_test)

#Dicionario de métricas e metadados
SVM_dict_v1 = {'Modelo':'SVM',
               'Versão':'1',
               'Kernel':'Linear',
               'Precision':precision_score(predictions_v1, y_test),
               'Recall':recall_score(predictions_v1, y_test),
               'F1':f1_score(predictions_v1, y_test),
               'Accuracy':accuracy_score(predictions_v1, y_test),
               'AUC':roc_auc_score(predictions_v1, y_test)}

#Print
print("Métricas em Teste:\n",SVM_dict_v1,"\nMatriz de Confusão:\n",confusion_matrix(predictions_v1, y_test))


#Modelo de Kernel Linear com os dados padronizados
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

model_v2 = svm.SVC(kernel = 'linear')
start = time.time()
model_v2.fit(X_train_scaled, y_train)
end = time.time()
print("Duração do treino do Modelo", end - start)

#Previsões
predictions_v2 = model_v2.predict(X_test_scaled)

#Dicionario de métricas e metadados
SVM_dict_v2 = {'Modelo':'SVM',
               'Versão':'2',
               'Kernel':'Linear',
               'Precision':precision_score(predictions_v2, y_test),
               'Recall':recall_score(predictions_v2, y_test),
               'F1':f1_score(predictions_v2, y_test),
               'Accuracy':accuracy_score(predictions_v2, y_test),
               'AUC':roc_auc_score(predictions_v2, y_test)}

#Print
print("Métricas em Teste:\n",SVM_dict_v2,"\nMatriz de Confusão:\n",confusion_matrix(predictions_v2, y_test))


#Modelo com Kernel RBF
model_v3 = svm.SVC(kernel = 'rbf')
C_range = np.array([50., 100., 200.])
gamma_range = np.array([0.3*0.001, 0.001, 3*0.001])

#Grid de hiperparâmetros
svm_param_grid = dict(gamma = gamma_range, C = C_range)

#Grid Search
start = time.time()
model_v3_grid_search_rbf = GridSearchCV(model_v3, svm_param_grid, cv=3)

#Treinamento
model_v3_grid_search_rbf.fit(X_train_scaled, y_train)
end = time.time()
print('\nDuração do Treinamento do Modelo RBF:', end - start)

#Acurácia em Treino
print(f"\nAcurácia em treinamento: {model_v3_grid_search_rbf.best_score_:.2%}")
print('')
print(f"Hiperparâmetros Ideais: {model_v3_grid_search_rbf.best_params_}")

#Previsões
predictions_v3 = model_v3_grid_search_rbf.predict(X_test_scaled)


#Dicionario de métricas e metadados
SVM_dict_v3 = {'Modelo':'SVM',
               'Versão':'3',
               'Kernel':'RBF',
               'Precision':precision_score(predictions_v3, y_test),
               'Recall':recall_score(predictions_v3, y_test),
               'F1':f1_score(predictions_v3, y_test),
               'Accuracy':accuracy_score(predictions_v3, y_test),
               'AUC':roc_auc_score(predictions_v3, y_test)}

#Print
print("Métricas em Teste:\n",SVM_dict_v3,"\nMatriz de Confusão:\n",confusion_matrix(predictions_v3, y_test))


#Modelo com Kernel Polinomial
model_v4 = svm.SVC(kernel = 'poly')

#Valores para o grid
r_range = np.array([0.5, 1])
gamma_range = np.array([0.1, 0.01, 0.001, 0.0001])
d_range = np.array([2, 3, 4])

#Grid de Hiperparâmetros
param_grid_poly = dict(gamma = gamma_range, degree = d_range, coef0 = r_range)

#Grid Search
start = time.time()
model_v4_grid_search_poly = GridSearchCV(model_v4, param_grid_poly, cv=3)

#Treinamento
model_v4_grid_search_poly.fit(X_train_scaled, y_train)
end = time.time()
print('\nDuração do Treinamento do Modelo Polinomial:', end - start)

#Acurácia em Treino
print(f'\nAcurácia em Treinamento: {model_v4_grid_search_poly.best_score_:.2%}')
print('')
print(f"Hiperparâmetros Ideais: {model_v4_grid_search_poly.best_params_}")

#Previsões
predictions_v4 = model_v4_grid_search_poly.predict(X_test_scaled)

#Dicionario de métricas e metadados
SVM_dict_v4 = {'Modelo':'SVM',
               'Versão':'4',
               'Kernel':'Poly',
               'Precision':precision_score(predictions_v4, y_test),
               'Recall':recall_score(predictions_v4, y_test),
               'F1':f1_score(predictions_v4, y_test),
               'Accuracy':accuracy_score(predictions_v4, y_test),
               'AUC':roc_auc_score(predictions_v4, y_test)}

#Print
print("Métricas em Teste:\n",SVM_dict_v4,"\nMatriz de Confusão:\n",confusion_matrix(predictions_v4, y_test))


#Concatenando todos os dicionários em um dataframe do Pandas
resumo = pd.DataFrame({'SVM_Model_1':pd.Series(SVM_dict_v1),
                      'SVM_Model_2':pd.Series(SVM_dict_v2),
                      'SVM_Model_3':pd.Series(SVM_dict_v3),
                      'SVM_Model_4':pd.Series(SVM_dict_v4)})

#Print
print('\n', resumo)

#Transpondo o Dataframe
df_plot = resumo.T

#Selecionando as métricas
metricas = ['Precision', 'Recall', 'F1', 'Accuracy', 'AUC']

#Gráficos
df_plot[metricas].plot(kind='bar', figsize=(10,6))
plt.title("Comparação de Métricas dos Modelos SVM")
plt.ylabel("Valor")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Métricas")
plt.tight_layout()
plt.savefig("comparacao_modelos_svm.png", dpi=300)
plt.show()