# split_creditcard.py (versão para chunks de 10 MB)
import pandas as pd
import os

print("A dividir o dataset creditcard.csv em chunks de ~10MB...")

# Definir número de linhas por chunk para ter arquivos de cerca de 10MB
linhas_por_chunk = 15000

# Ler o CSV
df = pd.read_csv('creditcard.csv')

# Calcular número de chunks
total_linhas = len(df)
num_chunks = (total_linhas // linhas_por_chunk) + 1

print(f"Total de linhas: {total_linhas}")
print(f"Linhas por chunk: {linhas_por_chunk}")
print(f"Número de chunks: {num_chunks}")

# Dividir e guardar
chunks_criados = []
for i in range(num_chunks):
    inicio = i * linhas_por_chunk
    fim = min((i + 1) * linhas_por_chunk, total_linhas)
    
    chunk = df.iloc[inicio:fim]
    nome_arquivo = f'creditcard_part_{i+1:03d}.csv'
    chunk.to_csv(nome_arquivo, index=False)
    
    tamanho_mb = os.path.getsize(nome_arquivo) / (1024 * 1024)
    chunks_criados.append((nome_arquivo, tamanho_mb))
    print(f"Criado: {nome_arquivo} - {len(chunk)} linhas - {tamanho_mb:.2f} MB")

print(f"\n✅ Dataset dividido em {num_chunks} chunks de ~10MB cada!")
print("\n📊 Sumário:")
for nome, tamanho in chunks_criados:
    print(f"   {nome}: {tamanho:.2f} MB")