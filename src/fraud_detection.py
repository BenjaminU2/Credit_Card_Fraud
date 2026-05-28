"""
Detecção de Fraudes em Transações Digitais
Projecto Final — Machine Learning
Dataset: Credit Card Fraud Detection (Kaggle)
"""

import os
import glob
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# CONFIGURAÇÕES DE PATH (AUTOMÁTICO)
# ─────────────────────────────────────────────
import os

# Descobre o diretório raiz do projeto (um nível acima de src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)   

# Define o caminho para os dados
RANDOM_STATE  = 42
TRAIN_RATIO   = 0.8
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'creditcard_part_*.csv')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'outputs')

# ─────────────────────────────────────────────
# 1. CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────
def carregar_dados(path, train_ratio):
    ficheiros = sorted(glob.glob(path))
    if not ficheiros:
        raise FileNotFoundError(f'Nenhum ficheiro encontrado em: {path}')

    random.shuffle(ficheiros)
    split = int(len(ficheiros) * train_ratio)

    treino_files = ficheiros[:split]
    teste_files  = ficheiros[split:]

    df_train = pd.concat([pd.read_csv(f) for f in treino_files], ignore_index=True)
    df_test  = pd.concat([pd.read_csv(f) for f in teste_files],  ignore_index=True)

    print('=' * 55)
    print('📂 CARREGAMENTO DOS DADOS')
    print('=' * 55)
    print(f'Total de chunks : {len(ficheiros)}')
    print(f'Chunks treino   : {len(treino_files)} → {df_train.shape[0]:,} linhas')
    print(f'Chunks teste    : {len(teste_files)}  → {df_test.shape[0]:,} linhas')
    print(f'Features        : {df_train.shape[1] - 1}')
    print()

    return df_train, df_test


# ─────────────────────────────────────────────
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# ─────────────────────────────────────────────
def eda(df_train):
    print('=' * 55)
    print('📊 ANÁLISE EXPLORATÓRIA')
    print('=' * 55)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    counts = df_train['Class'].value_counts()
    pct    = counts[1] / len(df_train) * 100

    print(f'Legítimas    : {counts[0]:,} ({100 - pct:.2f}%)')
    print(f'Fraudulentas : {counts[1]:,} ({pct:.4f}%)')
    print(f'Ratio        : 1 fraude por {counts[0] // counts[1]:,} legítimas')
    print()
    print('Estatísticas do Amount por classe:')
    print(df_train.groupby('Class')['Amount'].describe().round(2))
    print()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Distribuição das classes
    axes[0].bar(['Legítima (0)', 'Fraude (1)'], counts.values,
                color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[0].set_title('Distribuição das Classes', fontweight='bold')
    axes[0].set_ylabel('Nº de Transações')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

    # Amount por classe
    df_train[df_train['Class'] == 0]['Amount'].hist(
        bins=60, ax=axes[1], alpha=0.6, color='#2ecc71', label='Legítima')
    df_train[df_train['Class'] == 1]['Amount'].hist(
        bins=60, ax=axes[1], alpha=0.8, color='#e74c3c', label='Fraude')
    axes[1].set_xlim(0, 500)
    axes[1].set_title('Distribuição do Valor (Amount)', fontweight='bold')
    axes[1].set_xlabel('Valor (€)')
    axes[1].legend()

    plt.suptitle('EDA — Visão Geral do Dataset de Treino', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/eda_geral.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('✅ Gráfico EDA guardado em outputs/eda_geral.png\n')


# ─────────────────────────────────────────────
# 3. PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
def preprocessar(df_train, df_test):
    print('=' * 55)
    print('⚙️  PRÉ-PROCESSAMENTO')
    print('=' * 55)

    scaler = StandardScaler()

    for df in [df_train, df_test]:
        df['Amount'] = scaler.fit_transform(df[['Amount']])
        df['Time']   = scaler.fit_transform(df[['Time']])

    X_train = df_train.drop('Class', axis=1)
    y_train = df_train['Class']

    X_test  = df_test.drop('Class', axis=1)
    y_test  = df_test['Class']

    print(f'Treino : {X_train.shape[0]:,} amostras | Fraudes: {y_train.sum():,}')
    print(f'Teste  : {X_test.shape[0]:,} amostras  | Fraudes: {y_test.sum():,}')
    print()

    return X_train, y_train, X_test, y_test

# ─────────────────────────────────────────────
# 4. SMOTE — BALANCEAMENTO
# ─────────────────────────────────────────────
def aplicar_smote(X_train, y_train):
    print('=' * 55)
    print('⚖️  SMOTE — BALANCEAMENTO DE CLASSES')
    print('=' * 55)
    print(f'Antes  → Legítimas: {(y_train==0).sum():,} | Fraudes: {(y_train==1).sum():,}')

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f'Depois → Legítimas: {(y_res==0).sum():,} | Fraudes: {(y_res==1).sum():,}')
    print()

    return X_res, y_res


# ─────────────────────────────────────────────
# 5. TREINO DOS MODELOS
# ─────────────────────────────────────────────
def treinar_modelos(X_train_res, y_train_res, X_test, y_test):
    print('=' * 55)
    print('🚀 TREINO DOS MODELOS')
    print('=' * 55)

    modelos = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            eval_metric='logloss', verbosity=0,
            scale_pos_weight=(y_train_res == 0).sum() / (y_train_res == 1).sum()
        ),
    }

    resultados = {}

    for nome, modelo in modelos.items():
        print(f'  ⏳ {nome}...', end=' ', flush=True)
        modelo.fit(X_train_res, y_train_res)

        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

        resultados[nome] = {
            'modelo'   : modelo,
            'y_pred'   : y_pred,
            'y_prob'   : y_prob,
            'roc_auc'  : roc_auc_score(y_test, y_prob),
            'f1'       : f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall'   : recall_score(y_test, y_pred),
            'avg_prec' : average_precision_score(y_test, y_prob),
        }
        print(f'✅ ROC-AUC = {resultados[nome]["roc_auc"]:.4f}')

    print()
    return resultados


# ─────────────────────────────────────────────
# 6. AVALIAÇÃO
# ─────────────────────────────────────────────
def avaliar(resultados, y_test, X_train):
    print('=' * 55)
    print('📋 RELATÓRIOS DE CLASSIFICAÇÃO')
    print('=' * 55)

    for nome, res in resultados.items():
        print(f'\n--- {nome} ---')
        print(classification_report(
            y_test, res['y_pred'],
            target_names=['Legítima', 'Fraude']
        ))

    # Tabela comparativa
    tabela = pd.DataFrame([
        {
            'Modelo'              : nome,
            'ROC-AUC'             : round(res['roc_auc'], 4),
            'F1 (Fraude)'         : round(res['f1'], 4),
            'Precision'           : round(res['precision'], 4),
            'Recall'              : round(res['recall'], 4),
            'Avg Precision'       : round(res['avg_prec'], 4),
            'FN (Fraudes perdidas)': confusion_matrix(y_test, res['y_pred'])[1, 0],
        }
        for nome, res in resultados.items()
    ]).set_index('Modelo')

    print('=' * 70)
    print('📊 TABELA COMPARATIVA')
    print('=' * 70)
    print(tabela.to_string())
    print()

    melhor = tabela['ROC-AUC'].idxmax()
    print(f'🏆 Melhor modelo por ROC-AUC: {melhor} ({tabela.loc[melhor, "ROC-AUC"]})')
    print()

    # Matrizes de confusão
    fig, axes = plt.subplots(1, len(resultados), figsize=(16, 5))
    for ax, (nome, res) in zip(axes, resultados.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='RdYlGn',
                    xticklabels=['Legítima', 'Fraude'],
                    yticklabels=['Legítima', 'Fraude'])
        ax.set_title(f'{nome}\nROC-AUC: {res["roc_auc"]:.4f}', fontweight='bold')
        ax.set_ylabel('Real')
        ax.set_xlabel('Previsto')
    plt.suptitle('Matrizes de Confusão', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/matrizes_confusao.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Curvas ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    cores = ['#3498db', '#2ecc71', '#e74c3c']
    for (nome, res), cor in zip(resultados.items(), cores):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, color=cor, linewidth=2,
                label=f'{nome} (AUC = {res["roc_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Aleatório')
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curvas ROC', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/curvas_roc.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Importância das features
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, nome in zip(axes, ['Random Forest', 'XGBoost']):
        modelo = resultados[nome]['modelo']
        imp = pd.Series(modelo.feature_importances_,
                        index=X_train.columns).sort_values(ascending=False).head(15)
        imp.plot(kind='barh', ax=ax, color='#9b59b6', edgecolor='black')
        ax.set_title(f'Top 15 Features — {nome}', fontweight='bold')
        ax.invert_yaxis()
    plt.suptitle('Importância das Features', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('✅ Todos os gráficos guardados em outputs/')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    df_train, df_test         = carregar_dados(DATA_PATH, TRAIN_RATIO)
    eda(df_train)
    X_train, y_train, X_test, y_test = preprocessar(df_train, df_test)
    X_train_res, y_train_res  = aplicar_smote(X_train, y_train)
    resultados                = treinar_modelos(X_train_res, y_train_res, X_test, y_test)
    avaliar(resultados, y_test, X_train)

fraudes = df_train[df_train['Class'] == 1]['Amount']
print(f"Valores mínimos: {fraudes.min()}")
print(f"Valores máximos: {fraudes.max()}")
print(f"Média: {fraudes.mean():.2f}")

