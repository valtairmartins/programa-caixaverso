# #########################################################################
# ###################### VALTAIR MARTINS DE OLIVEIRA
# #########################################################################

# Importação + Limpeza Automática + Otimização
import pandas as pd
import numpy as np

def load_and_clean_csv(data):
    
    # 1. Carregar os dados
    if isinstance(data, str):
        df = pd.read_csv(data, low_memory=False)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("A entrada deve ser um caminho de arquivo (string) ou um DataFrame do pandas.")

    # 2. Remover colunas com mais de 40% de valores nulos
    df = df.loc[:, df.isnull().mean() < 0.4]

    # 3. Limpeza de strings
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.lower().replace("nan", "desconhecido")

    # 4. Preencher nulos em colunas numéricas com a mediana
    num_cols = df.select_dtypes(include=np.number).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # 5. Truncar outliers usando o método IQR
    for c in num_cols:
        Q1, Q3 = df[c].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df[c] = np.clip(df[c], a_min=limite_inferior, a_max=limite_superior)

    # 6. Otimizar tipos de dados para reduzir o consumo de memória
    for c in df.select_dtypes(include=np.number).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("category")

    return df

# #########################################################################
# ######################
# #########################################################################

import pandas as pd
import matplotlib.pyplot as plt

def analisar_taxa_juros(csv_path, coluna='int_rate', bins=30):
    
    # Importa o CSV
    df = pd.read_csv(csv_path)
    
    # Shape e estatísticas
    print(f"Shape do dataset: {df.shape}\n")
    print(f"Estatísticas da coluna '{coluna}':\n{df[coluna].describe()}\n")
    
    # Configurações do plot
    plt.style.use('default')  # Usa o estilo padrão do matplotlib
    fig, ax = plt.subplots(figsize=(8,4))
    
    # Cor do fundo do gráfico
    ax.set_facecolor('#f0f0f0')
    
    # Histograma
    ax.hist(df[coluna], bins=bins, color='#66c2a5', edgecolor='white')
    
    # Título
    ax.set_title(f"Distribuição de '{coluna}'", fontsize=14, pad=20, color='#333333')
    
    # Rótulos dos eixos
    ax.set_xlabel(f"{coluna}", color='#333333')
    ax.set_ylabel("Frequência", color='#333333')
    
    # Cores e transparência da grade
    ax.grid(color='#cccccc', linestyle='-', linewidth=0.5, alpha=0.8)
    
    # Cores dos ticks e bordas
    ax.tick_params(axis='x', colors='#333333')
    ax.tick_params(axis='y', colors='#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')

    plt.tight_layout()
    plt.show()
    
    return df

# #########################################################################
# ######################
# #########################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_top_correlations_heatmap(df, target_column='int_rate', top_n_percent=0.2):
    
    # Verifica se a coluna target existe e se é numérica
    if target_column not in df.columns or not pd.api.types.is_numeric_dtype(df[target_column]):
        print(f"Erro: A coluna '{target_column}' não existe ou não é numérica.")
        return

    # Calcula as correlações com a coluna target, excluindo-a
    correlacoes = df.corr(numeric_only=True)[target_column].abs().sort_values(ascending=False)
    correlacoes = correlacoes.drop(target_column)

    # Seleciona as colunas com a melhor correlação com base na porcentagem
    num_cols_to_plot = int(len(correlacoes) * top_n_percent)
    if num_cols_to_plot < 1:
        print(f"Não há colunas suficientes para plotar. Tente aumentar 'top_n_percent'.")
        return
        
    colunas_para_mapa = correlacoes.head(num_cols_to_plot).index.tolist()
    
    # Adiciona a coluna target de volta para o mapa de calor
    colunas_para_mapa.insert(0, target_column)

    # Define a paleta de cores personalizada
    cores_caixa = ['#B26F9B', '#005CA9', '#54BBAB']
    cmap_caixa = mcolors.LinearSegmentedColormap.from_list("caixa_cmap", cores_caixa)

    # Cria e plota o mapa de calor
    plt.figure(figsize=(10, 5))
    
    sns.heatmap(df[colunas_para_mapa].corr(), annot=True, fmt=".2f", cmap=cmap_caixa)
    
    plt.title(f'Correlação das Top {int(top_n_percent*100)}% Colunas da Lending Club')
    plt.show()

# #########################################################################
# ######################
# #########################################################################
     
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import shap
import re

# Verifica a importação de XGBoost e LightGBM
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from category_encoders import TargetEncoder as SimpleTargetEncoder
    HAS_TARGET_ENCODER = True
except ImportError:
    HAS_TARGET_ENCODER = False

def regression_modeling_pipeline(
    df: pd.DataFrame,
    target_col: str,
    encode_method: str = 'auto',
    use_pca: bool = False,
    pca_variance: float = 0.95,
    test_size: float = 0.2,
    cv: int = 3,
    n_iter_search: int = 5,
    random_state: int = 42,
    sample_size: int = 5000
):
    """
    Pipeline de modelagem de regressão com otimização de hiperparâmetros.
    Versão leve para teste rápido.
    """

    df = df.sample(n=min(sample_size, len(df)), random_state=random_state).reset_index(drop=True)

    if target_col not in df.columns:
        raise ValueError("target_col não encontrado no DataFrame")

    X = df.drop(columns=[target_col]).reset_index(drop=True)
    y = df[target_col].reset_index(drop=True)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if encode_method == 'auto':
        encode_method = 'onehot' if all(X[c].nunique() <= 10 for c in cat_cols) else ('target' if HAS_TARGET_ENCODER else 'label')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    print(f">>> Dados: treino {X_train.shape}, teste {X_test.shape}")
    print(f"Encoding escolhido: {encode_method}\n")

    transformers = []
    if len(num_cols) > 0:
        transformers.append(('num', StandardScaler(), num_cols))

    if encode_method == 'onehot':
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        preprocessor = ColumnTransformer(transformers + [('cat', cat_transformer, cat_cols)], remainder='drop')
        pipeline_base = Pipeline([('pre', preprocessor)])
    elif encode_method == 'label':
        cat_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        preprocessor = ColumnTransformer(transformers + [('cat', cat_transformer, cat_cols)], remainder='drop')
        pipeline_base = Pipeline([('pre', preprocessor)])
    elif encode_method == 'target':
        pipeline_base = None
    else:
        preprocessor = ColumnTransformer(transformers, remainder='drop') if transformers else None
        pipeline_base = Pipeline([('pre', preprocessor)]) if preprocessor else Pipeline([])

    estimators = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=4, n_jobs=-1, random_state=random_state)
    }

    # Definição dos parâmetros de busca
    param_distributions = {
        "RandomForest": {
            'n_estimators': [100, 200, 500, 1000, 1500],
            'max_depth': [3, 5, 8, 10, None],
            'min_samples_leaf': [1, 2, 4, 8, 16],
            'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9],
        }
    }

    if HAS_XGB:
        estimators["XGBoost"] = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.001,
                                             reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1,
                                             random_state=random_state, verbosity=0)
        param_distributions["XGBoost"] = {
            'n_estimators': [100, 500, 1000, 2000], 
            'max_depth': [3, 5, 7, 9, 12],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.001, 0.01, 0.1, 1, 10], 
            'reg_lambda': [0, 0.001, 0.01, 0.1, 1, 10]
        }

    if HAS_LGBM:
        estimators["LightGBM"] = LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=80, max_depth=4,
            reg_alpha=1, reg_lambda=1, bagging_fraction=0.8,
            feature_fraction=0.8, bagging_freq=1, n_jobs=-1, random_state=random_state
        )
        param_distributions["LightGBM"] = {
            'n_estimators': [500, 1000, 2000, 4000],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127, 255],
            'max_depth': [5, 8, 10, 15],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.001, 0.01, 0.1, 1, 10],
            'reg_lambda': [0, 0.001, 0.01, 0.1, 1, 10]
        }

    results = {}
    trained_estimators = {}

    for name, est in estimators.items():
        print(f"-> Buscando hiperparâmetros para {name} ...")
        steps = []
        if pipeline_base is not None:
            steps.append(('pre', pipeline_base.named_steps['pre']))
        if use_pca and (len(num_cols) + len(cat_cols)) > 0:
            steps.append(('pca', PCA(n_components=pca_variance, svd_solver='full', random_state=random_state)))
        steps.append(('est', est))
        pipe = Pipeline(steps)
        pdist = {f'est__{k}': v for k, v in param_distributions[name].items()}
        rs = RandomizedSearchCV(pipe, param_distributions=pdist, n_iter=n_iter_search,
                                scoring='neg_root_mean_squared_error',
                                cv=KFold(n_splits=cv, shuffle=True, random_state=random_state),
                                random_state=random_state, n_jobs=-1)
        rs.fit(X_train, y_train)
        best_pipe = rs.best_estimator_
        trained_estimators[name] = best_pipe

        y_tr_pred = best_pipe.predict(X_train)
        y_te_pred = best_pipe.predict(X_test)
        results[name] = {
            "metrics": {
                "rmse_tr": np.sqrt(mean_squared_error(y_train, y_tr_pred)),
                "r2_tr": r2_score(y_train, y_tr_pred),
                "rmse_te": np.sqrt(mean_squared_error(y_test, y_te_pred)),
                "r2_te": r2_score(y_test, y_te_pred)
            }
        }

    if len(trained_estimators) >= 2:
        print("-> Treinando StackingRegressor ...")
        base_estimators = [(name, est) for name, est in trained_estimators.items()]
        meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
        
        stack = StackingRegressor(estimators=base_estimators, final_estimator=meta, n_jobs=-1)
        stack.fit(X_train, y_train)
        
        trained_estimators["Stacking"] = stack
        
        y_tr_pred = stack.predict(X_train)
        y_te_pred = stack.predict(X_test)
        results["Stacking"] = {
            "metrics": {
                "rmse_tr": np.sqrt(mean_squared_error(y_train, y_tr_pred)),
                "r2_tr": r2_score(y_train, y_tr_pred),
                "rmse_te": np.sqrt(mean_squared_error(y_test, y_te_pred)),
                "r2_te": r2_score(y_test, y_te_pred)
            }
        }
    
    # Exibir os melhores hiperparâmetros
    print("\n=== Melhores Hiperparâmetros Escolhidos ===")
    best_params_dict = {}
    for name, pipeline in trained_estimators.items():
        if name != "Stacking":
            estimator = pipeline.named_steps.get('est')
            if estimator:
                params = estimator.get_params()
                optimized_params = {k: v for k, v in params.items() if k in param_distributions[name]}
                best_params_dict[name] = optimized_params
        else:
            final_est = pipeline.final_estimator
            best_params_dict[name] = {"Final Estimator": final_est.__class__.__name__}
            
    df_params = pd.DataFrame(best_params_dict).T.fillna("-")
    print(df_params)

    df_results = pd.DataFrame({
        model: {
            "RMSE_treino": info["metrics"]["rmse_tr"],
            "R2_treino": info["metrics"]["r2_tr"],
            "RMSE_teste": info["metrics"]["rmse_te"],
            "R2_teste": info["metrics"]["r2_te"]
        }
        for model, info in results.items()
    }).T

    df_results = df_results.sort_values(by="RMSE_teste", ascending=True)

    print("\n=== Ranking Final de Modelos ===")
    print(df_results.round(6))

    return {
        'results': results,
        'ranking': df_results,
        'trained_estimators': trained_estimators,
        'X_test': X_test,
        'X_train': X_train,
        'y_train': y_train,
        'y_test': y_test
    }


# #########################################################################
# ######################
# #########################################################################

   
import pandas as pd
import matplotlib.pyplot as plt

def plot_model_ranking(df: pd.DataFrame):
    
    # --- Definição de cores ---
    COR_AZUL_MARINHO = "#004198"
    COR_TURQUESA = "#54BBAB"
    COR_VERMELHA = "#A50026"

    color_map = {
        'Stacking': COR_AZUL_MARINHO,
        'LightGBM': COR_TURQUESA,
        'XGBoost': COR_TURQUESA,
        'RandomForest': COR_VERMELHA
    }
    colors = [color_map[m] for m in df['Modelo']]

    # --- Criação do gráfico ---
    plt.figure(figsize=(11, 6))
    plt.scatter(df['RMSE_teste'], df['R2_teste'], s=150, color=colors, alpha=0.7)

    # --- Offsets personalizados ---
    text_params = {
        'Stacking':     {'xytext': (-15, 10), 'ha': 'left',   'va': 'bottom'},
        'LightGBM':     {'xytext': (0, 20),   'ha': 'center', 'va': 'bottom'},
        'XGBoost':      {'xytext': (0, -20),  'ha': 'center', 'va': 'top'},
        'RandomForest': {'xytext': (-5, -5),  'ha': 'right',  'va': 'top'}
    }

    for i, modelo in enumerate(df['Modelo']):
        params = text_params.get(modelo, {'xytext': (0, 10), 'ha': 'center', 'va': 'bottom'})
        plt.annotate(
            f"{modelo}\nRMSE: {df['RMSE_teste'][i]:.3f}\nR²: {df['R2_teste'][i]:.3f}",
            (df['RMSE_teste'][i], df['R2_teste'][i]),
            textcoords="offset points",
            xytext=params['xytext'],
            ha=params['ha'],
            va=params['va'],
            fontsize=10,
            color=COR_AZUL_MARINHO,
            fontweight='bold' if modelo == 'Stacking' else 'normal'
        )

    # --- Ajustes de layout ---
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Otimizando a Decisão: Equilíbrio entre Erro e Precisão", fontsize=14, fontweight='bold')
    plt.xlabel("RMSE (Erro - quanto menor, melhor)", fontsize=12)
    plt.ylabel("R² (Precisão - quanto maior, melhor)", fontsize=12)
    plt.gca().invert_xaxis()

    min_rmse, max_rmse = df['RMSE_teste'].min(), df['RMSE_teste'].max()
    min_r2, max_r2 = df['R2_teste'].min(), df['R2_teste'].max()
    plt.xlim(max_rmse * 1.05, min_rmse * 0.95)
    plt.ylim(min_r2 * 0.98, max_r2 * 1.02)

    plt.show()

# #########################################################################
# ######################
# #########################################################################
    