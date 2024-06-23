import os
import zipfile
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, recall_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential

# Função para descompactar o arquivo ARFF
def descompactar_arquivo(input_zip, output_file):
    with zipfile.ZipFile(input_zip, 'r') as zf:
        zf.extractall()
        zf.extract('dataset.arff', path=output_file)

# Descompactar o arquivo ARFF
descompactar_arquivo('dataset.zip', '.')

# Caminho do arquivo ARFF
file_path = 'dataset.arff'

# Criar pasta de resultados
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Carregar o arquivo ARFF
data, meta = arff.loadarff(file_path)

# Converter para DataFrame pandas
df = pd.DataFrame(data)

# Decodificar os dados categóricos (se houver)
for column in df.select_dtypes([object]):
    df[column] = df[column].str.decode('utf-8')

# Calcular a correlação de todas as variáveis com a variável alvo 'Class'
correlation_matrix = df.corr()
correlations_with_class = correlation_matrix['Class'].abs().sort_values(ascending=False)

# Selecionar as variáveis com a maior correlação (top 10)
top_features = correlations_with_class.index[1:11]  # Ignorar a primeira entrada que é a própria 'Class'
print("Principais características para o treinamento:", top_features)

# Plotar as correlações das top features com a variável 'Class'
plt.figure(figsize=(10, 6))
sns.barplot(x=correlations_with_class[top_features], y=top_features, palette='viridis')
plt.xlabel('Correlação Absoluta com Class')
plt.ylabel('Features')
plt.title('Correlação das Principais Features com a Variável Class')
plt.savefig(os.path.join(results_dir, 'correlation_with_class.png'))
plt.show()

# Criar um DataFrame com as variáveis selecionadas para treinar o modelo
df_selected_features = df[top_features].copy()
df_selected_features.loc[:, 'Class'] = df['Class']  # Adicionar a variável alvo

# Dividir os dados em conjuntos de treino e teste
X = df_selected_features.drop('Class', axis=1)
y = df_selected_features['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar técnicas de balanceamento
# SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Random Oversampling
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

# Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

def train_neural_network(X_train, y_train, X_test, y_test):
    # Criar o modelo de rede neural
    nn_model = Sequential()
    nn_model.add(Input(shape=(X_train.shape[1],)))
    nn_model.add(Dense(32, activation='relu'))
    nn_model.add(Dense(16, activation='relu'))
    nn_model.add(Dense(1, activation='sigmoid'))
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)
    return nn_model

def train_and_evaluate(X_train, y_train, X_test, y_test, technique_name):
    nn_model = train_neural_network(X_train, y_train, X_test, y_test)

    # Previsões e probabilidades da rede neural
    nn_probs = nn_model.predict(X_test)
    nn_pred = (nn_probs > 0.5).astype(int)
    nn_auc = roc_auc_score(y_test, nn_probs)

    # Treinar os modelos de Random Forest, Gradient Boosting e Regressão Logística
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)

    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_probs = gb_model.predict_proba(X_test)[:, 1]
    gb_auc = roc_auc_score(y_test, gb_probs)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)

    # Avaliar o desempenho dos modelos
    metrics = {
        'Model': ['Neural Network', 'Random Forest', 'Gradient Boosting', 'Logistic Regression'],
        'Accuracy': [accuracy_score(y_test, nn_pred),
                     accuracy_score(y_test, rf_pred),
                     accuracy_score(y_test, gb_pred),
                     accuracy_score(y_test, lr_pred)],
        'F1-Score': [f1_score(y_test, nn_pred),
                     f1_score(y_test, rf_pred),
                     f1_score(y_test, gb_pred),
                     f1_score(y_test, lr_pred)],
        'Recall': [recall_score(y_test, nn_pred),
                   recall_score(y_test, rf_pred),
                   recall_score(y_test, gb_pred),
                   recall_score(y_test, lr_pred)],
        'AUC': [nn_auc, rf_auc, gb_auc, lr_auc]
    }

    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)

    # Salvar as métricas em um arquivo CSV
    metrics_df.to_csv(os.path.join(results_dir, f'metrics_{technique_name}.csv'), index=False)

    # Plotar as métricas
    metrics_df.set_index('Model', inplace=True)
    metrics_df.plot(kind='bar', figsize=(14, 8), colormap='viridis')
    plt.title(f'Métricas de Desempenho - {technique_name}')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(results_dir, f'metrics_{technique_name}.png'))
    plt.show()

    # Plotar a Curva ROC
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='darkblue')
    plt.plot(*roc_curve(y_test, nn_probs)[:2], label=f'Neural Network (AUC = {nn_auc:.3f})')
    plt.plot(*roc_curve(y_test, rf_probs)[:2], label=f'Random Forest (AUC = {rf_auc:.3f})')
    plt.plot(*roc_curve(y_test, gb_probs)[:2], label=f'Gradient Boosting (AUC = {gb_auc:.3f})')
    plt.plot(*roc_curve(y_test, lr_probs)[:2], label=f'Logistic Regression (AUC = {lr_auc:.3f})')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC - {technique_name}')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'roc_curve_{technique_name}.png'))
    plt.show()

    return nn_model, rf_model, gb_model, lr_model

# Avaliar técnicas de balanceamento
nn_model_smote, rf_model_smote, gb_model_smote, lr_model_smote = train_and_evaluate(X_train_smote, y_train_smote, X_test, y_test, "SMOTE")

nn_model_ros, rf_model_ros, gb_model_ros, lr_model_ros = train_and_evaluate(X_train_ros, y_train_ros, X_test, y_test, "Random Oversampling")

nn_model_rus, rf_model_rus, gb_model_rus, lr_model_rus = train_and_evaluate(X_train_rus, y_train_rus, X_test, y_test, "Random Undersampling")

# Selecionar todas as instâncias da classe minoritária e um número equivalente da classe majoritária
fraudulent = df_selected_features[df_selected_features['Class'] == 1]
non_fraudulent = df_selected_features[df_selected_features['Class'] == 0].sample(n=len(fraudulent), random_state=42)

# Concatenar os dois DataFrames
test_subset = pd.concat([fraudulent, non_fraudulent])
test_subset = test_subset.sample(frac=1, random_state=42)  # Embaralhar o subset

# Separar as features e a classe verdadeira
X_test_subset = test_subset.drop('Class', axis=1)
y_test_subset = test_subset['Class']

# Previsão e comparação para SMOTE
test_subset['Classe Predita NN SMOTE'] = (nn_model_smote.predict(X_test_subset) >= 0.5).astype(float)
test_subset['Classe Predita RF SMOTE'] = rf_model_smote.predict(X_test_subset)
test_subset['Classe Predita GB SMOTE'] = gb_model_smote.predict(X_test_subset)
test_subset['Classe Predita LR SMOTE'] = lr_model_smote.predict(X_test_subset)

# Previsão e comparação para Random Oversampling
test_subset['Classe Predita NN ROS'] = (nn_model_ros.predict(X_test_subset) >= 0.5).astype(float)
test_subset['Classe Predita RF ROS'] = rf_model_ros.predict(X_test_subset)
test_subset['Classe Predita GB ROS'] = gb_model_ros.predict(X_test_subset)
test_subset['Classe Predita LR ROS'] = lr_model_ros.predict(X_test_subset)

# Previsão e comparação para Random Undersampling
test_subset['Classe Predita NN RUS'] = (nn_model_rus.predict(X_test_subset) >= 0.5).astype(float)
test_subset['Classe Predita RF RUS'] = rf_model_rus.predict(X_test_subset)
test_subset['Classe Predita GB RUS'] = gb_model_rus.predict(X_test_subset)
test_subset['Classe Predita LR RUS'] = lr_model_rus.predict(X_test_subset)

# Ajustar as configurações de impressão do pandas para exibir todas as colunas
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Exibir o subconjunto com as previsões e os valores reais
print("\nSubconjunto de dados de teste com previsões e valores reais:")
print(test_subset)

# Salvar o subconjunto em um arquivo CSV
test_subset.to_csv(os.path.join(results_dir, 'test_subset_predictions.csv'), index=False)

# Calcular a porcentagem de acertos de cada modelo
rf_accuracy_smote = accuracy_score(y_test_subset, test_subset['Classe Predita RF SMOTE'])
gb_accuracy_smote = accuracy_score(y_test_subset, test_subset['Classe Predita GB SMOTE'])
lr_accuracy_smote = accuracy_score(y_test_subset, test_subset['Classe Predita LR SMOTE'])
nn_accuracy_smote = accuracy_score(y_test_subset, test_subset['Classe Predita NN SMOTE'])

rf_accuracy_ros = accuracy_score(y_test_subset, test_subset['Classe Predita RF ROS'])
gb_accuracy_ros = accuracy_score(y_test_subset, test_subset['Classe Predita GB ROS'])
lr_accuracy_ros = accuracy_score(y_test_subset, test_subset['Classe Predita LR ROS'])
nn_accuracy_ros = accuracy_score(y_test_subset, test_subset['Classe Predita NN ROS'])

rf_accuracy_rus = accuracy_score(y_test_subset, test_subset['Classe Predita RF RUS'])
gb_accuracy_rus = accuracy_score(y_test_subset, test_subset['Classe Predita GB RUS'])
lr_accuracy_rus = accuracy_score(y_test_subset, test_subset['Classe Predita LR RUS'])
nn_accuracy_rus = accuracy_score(y_test_subset, test_subset['Classe Predita NN RUS'])

print(f"\nPorcentagem de acertos Random Forest (SMOTE): {rf_accuracy_smote * 100:.2f}%")
print(f"Porcentagem de acertos Gradient Boosting (SMOTE): {gb_accuracy_smote * 100:.2f}%")
print(f"Porcentagem de acertos Regressão Logística (SMOTE): {lr_accuracy_smote * 100:.2f}%")
print(f"Porcentagem de acertos Rede Neural (SMOTE): {nn_accuracy_smote * 100:.2f}%")

print(f"\nPorcentagem de acertos Random Forest (Random Oversampling): {rf_accuracy_ros * 100:.2f}%")
print(f"Porcentagem de acertos Gradient Boosting (Random Oversampling): {gb_accuracy_ros * 100:.2f}%")
print(f"Porcentagem de acertos Regressão Logística (Random Oversampling): {lr_accuracy_ros * 100:.2f}%")
print(f"Porcentagem de acertos Rede Neural (Random Oversampling): {nn_accuracy_ros * 100:.2f}%")

print(f"\nPorcentagem de acertos Random Forest (Random Undersampling): {rf_accuracy_rus * 100:.2f}%")
print(f"Porcentagem de acertos Gradient Boosting (Random Undersampling): {gb_accuracy_rus * 100:.2f}%")
print(f"Porcentagem de acertos Regressão Logística (Random Undersampling): {lr_accuracy_rus * 100:.2f}%")
print(f"Porcentagem de acertos Rede Neural (Random Undersampling): {nn_accuracy_rus * 100:.2f}%")

# Porcentagens de acertos
rf_accuracies = [rf_accuracy_smote, rf_accuracy_ros, rf_accuracy_rus]
gb_accuracies = [gb_accuracy_smote, gb_accuracy_ros, gb_accuracy_rus]
lr_accuracies = [lr_accuracy_smote, lr_accuracy_ros, lr_accuracy_rus]
nn_accuracies = [nn_accuracy_smote, nn_accuracy_ros, nn_accuracy_rus]

# Nomes dos modelos e das técnicas de amostragem
models = ['Random Forest', 'Gradient Boosting', 'Regressão Logística', 'Rede Neural']
sampling_techniques = ['SMOTE', 'Random Oversampling', 'Random Undersampling']

# Criar o gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Definir a largura das barras
bar_width = 0.15

# Posições das barras no eixo x
r1 = np.arange(len(sampling_techniques))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Plotar as barras
plt.bar(r1, rf_accuracies, color='b', width=bar_width, edgecolor='grey', label='Random Forest')
plt.bar(r2, gb_accuracies, color='g', width=bar_width, edgecolor='grey', label='Gradient Boosting')
plt.bar(r3, lr_accuracies, color='r', width=bar_width, edgecolor='grey', label='Regressão Logística')
plt.bar(r4, nn_accuracies, color='c', width=bar_width, edgecolor='grey', label='Rede Neural')

# Adicionar legendas e rótulos
plt.xlabel('Técnica de Amostragem', fontweight='bold')
plt.ylabel('Porcentagem de Acertos', fontweight='bold')
plt.title('Porcentagem de Acertos por Modelo e Técnica de Amostragem')
plt.xticks([r + bar_width * 1.5 for r in range(len(sampling_techniques))], sampling_techniques)
plt.legend()

# Salvar o gráfico
plt.savefig(os.path.join(results_dir, 'accuracies_by_sampling_technique.png'))
plt.show()

# Calcular as matrizes de confusão para cada modelo e técnica de amostragem
rf_cm_smote = confusion_matrix(y_test_subset, test_subset['Classe Predita RF SMOTE'])
gb_cm_smote = confusion_matrix(y_test_subset, test_subset['Classe Predita GB SMOTE'])
lr_cm_smote = confusion_matrix(y_test_subset, test_subset['Classe Predita LR SMOTE'])
nn_cm_smote = confusion_matrix(y_test_subset, test_subset['Classe Predita NN SMOTE'])

rf_cm_ros = confusion_matrix(y_test_subset, test_subset['Classe Predita RF ROS'])
gb_cm_ros = confusion_matrix(y_test_subset, test_subset['Classe Predita GB ROS'])
lr_cm_ros = confusion_matrix(y_test_subset, test_subset['Classe Predita LR ROS'])
nn_cm_ros = confusion_matrix(y_test_subset, test_subset['Classe Predita NN ROS'])

rf_cm_rus = confusion_matrix(y_test_subset, test_subset['Classe Predita RF RUS'])
gb_cm_rus = confusion_matrix(y_test_subset, test_subset['Classe Predita GB RUS'])
lr_cm_rus = confusion_matrix(y_test_subset, test_subset['Classe Predita LR RUS'])
nn_cm_rus = confusion_matrix(y_test_subset, test_subset['Classe Predita NN RUS'])

# Plotar as matrizes de confusão
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# Criar subplots para cada técnica de amostragem e modelo
for i, model in enumerate(models):
    axes[0, i].set_title(model)
    sns.heatmap(rf_cm_smote, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
    sns.heatmap(gb_cm_smote, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1], cbar=False)
    sns.heatmap(lr_cm_smote, annot=True, fmt='d', cmap='Reds', ax=axes[0, 2], cbar=False)
    sns.heatmap(nn_cm_smote, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 3], cbar=False)
    sns.heatmap(rf_cm_ros, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], cbar=False)
    sns.heatmap(gb_cm_ros, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1], cbar=False)
    sns.heatmap(lr_cm_ros, annot=True, fmt='d', cmap='Reds', ax=axes[1, 2], cbar=False)
    sns.heatmap(nn_cm_ros, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 3], cbar=False)
    sns.heatmap(rf_cm_rus, annot=True, fmt='d', cmap='Blues', ax=axes[2, 0], cbar=False)
    sns.heatmap(gb_cm_rus, annot=True, fmt='d', cmap='Greens', ax=axes[2, 1], cbar=False)
    sns.heatmap(lr_cm_rus, annot=True, fmt='d', cmap='Reds', ax=axes[2, 2], cbar=False)
    sns.heatmap(nn_cm_rus, annot=True, fmt='d', cmap='Oranges', ax=axes[2, 3], cbar=False)

# Ajustar layout
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'))
plt.show()