import pandas as pd # type: ignore
import numpy as np # type: ignore
import lightgbm as lgbm # type: ignore
from sklearn.model_selection import cross_val_score # type: ignore
from sklearn.metrics import make_scorer, recall_score # type: ignore

# Función para crear columnas con cada parte de una cadena de texto estilo mascara de red, creando estas partes por la separacion de la cadena por el punto.

def add_mask_features(df):
    mask_features=["EngineVersion", "AppVersion", "AvSigVersion", "OsBuildLab", "Census_OSVersion"]

    for feature in mask_features:
        split_df=df[feature].str.split(".", expand=True)

        split_df.columns = [f"{feature}_part_{i+1}" for i in range(split_df.shape[1])]
        
        df = pd.concat([df, split_df], axis=1)

    return df

# Función para crear columnas contextuales extras a partir de la media, el minimo y el maximo de una lista de columnas dada.

def generate_grouped_stats(df, num_cols, cat_cols):
    df_ext = df.copy()
    original_index = df_ext.index 
    added_cols = []
    
    for cat in cat_cols:
        group_by_feat = df_ext.groupby(by=[cat])
        for num_feat in num_cols:
            col_names = [cat, 
                        f'mean_{num_feat}_by_{cat}',
                        f'max_{num_feat}_by_{cat}', 
                        f'min_{num_feat}_by_{cat}']
            df_grouped = group_by_feat[num_feat].agg([np.mean, np.max, np.min]).reset_index() 
            df_grouped.columns = col_names

            added_cols += col_names[1:]
            df_ext = pd.merge(left=df_ext.reset_index(), right=df_grouped, how='left', on=[cat], suffixes=('', '_feat'))
            df_ext.set_index('index', inplace=True)

    df_ext.index = original_index
    return df_ext, added_cols


# Función para crear mas columnas contextuales a partir de las creadas en el anterior metodo.

def generate_synthetic_features(df, added_cols):
    df_ext = df.copy()
    for col in added_cols:
        parts=col.split('_')
        stat=parts[0]
        num_feat = '_'.join(parts[1:-2])
        cat = parts[-1]
        if stat == 'mean':        
            df_ext[num_feat + '_ratio_mean_on_' + cat] = round(df_ext[num_feat] / df_ext[col], 2)
        elif stat == 'max':
            df_ext[num_feat + '_amplitude_on_' + cat] = round(df_ext[col] - df_ext['min_' + num_feat + '_by_' + cat], 2)
            df_ext[num_feat + '_ratio_max_on_' + cat] = round(df_ext[num_feat] / df_ext[col], 2)
    
    return df_ext


# Función para realizar tuneo de hiperparametros con Optuna.

def objective(trial, X, y):

    # Grid de hiperparametros 
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 20, 100, step=5),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1, step=0.1),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.99, step=0.05),
        'num_leaves': trial.suggest_int('num_leaves', 30, 100, step=10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, step=0.05),
        'max_depth': trial.suggest_int('max_depth', 10, 100, step=5),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1, step=0.2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1, step=0.2),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1, step=0.2)
    }

    model = lgbm.LGBMClassifier(**params)

    scorer = make_scorer(recall_score)
    scores = cross_val_score(model, X, y, cv=5, scoring=scorer, n_jobs=-1)

    return scores.mean()