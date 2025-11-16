import pandas as pd
import numpy as np
from typing import Union
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score

# --- Совместимый OneHotEncoder для разных версий scikit-learn ---
# В новых версиях используется параметр `sparse_output`, в старых — `sparse`.
# Эта обёртка вернёт корректный OHE в любом случае.
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Для старых версий scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# --------------------------------------------------------------
# Лабораторная работа №5 — Метрические методы и решающие деревья
# Файл: Галиев_АА_lab_5.py
# Функция по спецификации:classification_training(data)
# Требования из методички:
#  - целевая переменная: 'mental_wellness_index_0_100' -> бинаризация (<15 -> 0; >=15 -> 1)
#  - обучить KNN и DecisionTree с осмысленным подбором гиперпараметров
#  - вывести метрики accuracy и f1 по шаблону: "{model_name}: {accuracy}; {f1_score}" (model_name: KNN, DT)
#  - построить дерево обученной модели решающего дерева (plot_tree)
# --------------------------------------------------------------


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Стрипуем пробелы/кавычки, удаляем Unnamed-столбцы, нормализуем пропуски."""
    df = df.copy()
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    to_drop = [c for c in df.columns if c.lower().startswith("unnamed")]
    if to_drop:
        df = df.drop(columns=to_drop)
    # Пустые строки и явные текстовые маркеры пропусков -> NaN
    df = df.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "None": np.nan})
    return df


def _find_target_column(df: pd.DataFrame) -> str:
    """Ищем столбец-мишень по точному или эвристическому совпадению."""
    exact = [c for c in df.columns if c == 'mental_wellness_index_0_100']
    if exact:
        return exact[0]
    fuzzy = [
        c for c in df.columns
        if ("mental" in c.lower()) and ("well" in c.lower()) and ("100" in c or "index" in c.lower())
    ]
    if fuzzy:
        return fuzzy[0]
    raise KeyError("Не найден столбец 'mental_wellness_index_0_100' в данных.")


def _prepare_xy(df: pd.DataFrame, target_col: str):
    y = pd.to_numeric(df[target_col], errors='coerce')
    y = (y >= 15).astype('float').astype('Int64')  # через float -> Int64 для маски
    # Уберём ID-поля и саму цель
    drop_cols = [target_col] + [c for c in df.columns if 'user_id' in c.lower()]
    X = df.drop(columns=drop_cols, errors='ignore')
    # Удалим строки, где цель отсутствует
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask].astype(int)
    return X, y


def _build_pipelines():
    # Выделим числовые/категориальные признаки
    num_sel = selector(dtype_include=["number"])
    cat_sel = selector(dtype_exclude=["number"])  # всё нечисловое — категориальное

    # Для KNN масштабирование важно (метрический алгоритм), категориальные — OHE
    pre_knn = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_sel),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", make_ohe())
            ]), cat_sel),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    knn = KNeighborsClassifier()
    knn_pipe = Pipeline([("preprocess", pre_knn), ("model", knn)])

    # Сетка умеренного размера, чтобы не перегружать вычисления, но удержать bias/variance баланс.
    # Обоснование:
    #  - n_neighbors: малые значения (3–7) лучше улавливают локальные структуры (риск переобучения),
    #    большие (9–11) сглаживают шум (риск недообучения). Подбираем компромисс.
    #  - weights: 'distance' часто устойчивее при шуме и разреженности.
    #  - p: 1 (Манхэттен) и 2 (Евклид) — стандартный выбор.
    param_grid_knn = {
        "model__n_neighbors": [3, 5, 7, 9, 11],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }

    # Для DT масштабирование не требуется, но категориальные кодируем OHE
    pre_dt = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_sel),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", make_ohe())
            ]), cat_sel),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    dt_pipe = Pipeline([("preprocess", pre_dt), ("model", dt)])

    # Обоснование гиперпараметров DT:
    #  - max_depth ограничивает сложность дерева и контролирует переобучение;
    #  - min_samples_leaf предотвращает слишком мелкие листья;
    #  - ccp_alpha — пост-прунинг по стоимости сложности, сглаживает переобучение.
    param_grid_dt = {
        "model__max_depth": [3, 5, 7, 9, None],
        "model__min_samples_leaf": [1, 5, 10],
        "model__ccp_alpha": [0.0, 0.001, 0.005],
    }

    return knn_pipe, param_grid_knn, dt_pipe, param_grid_dt


# --- helper: аккуратная отрисовка дерева внутри функции, без глобальных переменных ---
def _plot_dt_tree(best_pipe, title=None):
    if title is None:
        title = "Decision Tree — визуализация до глубины 3\nГалиев А.А., Лабораторная работа №5"
    preprocess = best_pipe.named_steps['preprocess']
    model = best_pipe.named_steps['model']
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = None

    plt.figure(figsize=(18, 12), facecolor='white')
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["<15", ">=15"],
        filled=True,
        max_depth=3,
        fontsize=10,
        rounded=True
    )
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color("#333333")
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Условия разбиений", fontsize=14, labelpad=10)
    plt.ylabel("Листовые предсказания", fontsize=14, labelpad=10)
    plt.tight_layout()
    plt.show()


def classification_training(data: Union[str, pd.DataFrame]) -> None:
    """
    Обучает KNN и DecisionTree на датасете из ЛР2.
    - Бинаризует 'mental_wellness_index_0_100' по порогу 15 (<15 -> 0; >=15 -> 1)
    - Печатает метрики accuracy и f1 по шаблону
    - Строит дерево обученной модели DT (plot_tree)
    Ничего не возвращает.
    """
    # 1) Загрузка данных
    if isinstance(data, str):
        # Попробуем автоматически распознать разделитель (',' или '\t')
        try:
            df = pd.read_csv(data, sep=None, engine='python')
        except Exception:
            # Иногда датасеты приходят как TSV
            df = pd.read_csv(data, sep='\t')
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Аргумент 'data' должен быть путем к файлу или pandas.DataFrame")

    # 2) Базовая чистка колонок
    df = _clean_columns(df)

    # 3) Целевая переменная + фичи
    target_col = _find_target_column(df)
    X, y = _prepare_xy(df, target_col)

    # 4) Трейн/тест
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5) Конвейеры и сетки
    knn_pipe, param_grid_knn, dt_pipe, param_grid_dt = _build_pipelines()

    # 6) Подбор KNN
    gs_knn = GridSearchCV(
        knn_pipe,
        param_grid_knn,
        scoring='f1',  # бинарная целевая — уместно оптимизировать F1
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    gs_knn.fit(X_train, y_train)

    y_pred_knn = gs_knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    f1_knn = f1_score(y_test, y_pred_knn)
    print(f"KNN: {acc_knn:.4f}; {f1_knn:.4f}")

    # 7) Подбор Decision Tree
    gs_dt = GridSearchCV(
        dt_pipe,
        param_grid_dt,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    gs_dt.fit(X_train, y_train)

    y_pred_dt = gs_dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    f1_dt = f1_score(y_test, y_pred_dt)
    print(f"DT: {acc_dt:.4f}; {f1_dt:.4f}")

    # 8) Визуализация дерева (внутри функции)
    _plot_dt_tree(gs_dt.best_estimator_)


# Точка входа: самостоятельный запуск файла
if __name__ == "__main__":
    # Укажи путь к своему датасету. По умолчанию — загрузки на Windows
    csv_path = r"C:\Users\Andrey\Downloads\DB_3 (1).csv"
    classification_training(csv_path)
