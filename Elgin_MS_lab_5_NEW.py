import logging
import zipfile
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Изменяю уровень логирования на ERROR.
# Теперь в консоль будут выводиться только сообщения об ошибках,
# а информационные логи (INFO) будут скрыты.
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')


def download_yandex_disk_file(url: str) -> bytes | None:
    """
    Загружает файл с публичной ссылки Яндекс.Диска.
    """
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    params = {'public_key': url}
    try:
        logging.info("Получение ссылки на скачивание архива...")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        download_url = response.json()['href']

        logging.info("Скачивание архива...")
        download_response = requests.get(download_url)
        download_response.raise_for_status()

        logging.info("Распаковка файла из архива в памяти...")
        zip_file = zipfile.ZipFile(BytesIO(download_response.content))
        csv_filename = next(
            (name for name in zip_file.namelist() if name.endswith('.csv')),
            None
        )

        if csv_filename:
            logging.info(f"Извлечение файла: {csv_filename}")
            csv_content = zip_file.read(csv_filename)
            logging.info("Файл успешно извлечен.")
            return csv_content
        else:
            logging.error("CSV файл не найден в архиве.")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка при загрузке файла: {e}")
        return None
    except zipfile.BadZipFile:
        logging.error(
            "Загруженный файл не является корректным ZIP-архивом."
        )
        return None


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет полную предобработку данных: очистку, заполнение пропусков,
    кодирование категориальных признаков и удаление выбросов.
    """
    logging.info("Начало предобработки данных...")
    df = df.copy()

    if 'Unnamed: 15' in df.columns:
        df = df.drop('Unnamed: 15', axis=1)
    df.columns = df.columns.str.replace('\t', '', regex=False)

    df = df.drop_duplicates(subset=['user_id'], keep='first')

    df.dropna(
        subset=['age', 'gender', 'occupation', 'work_mode'], inplace=True
    )
    df.reset_index(drop=True, inplace=True)

    numeric_cols_with_nan = [
        'screen_time_hours', 'work_screen_hours', 'leisure_screen_hours',
        'sleep_quality_1_5', 'stress_level_0_10',
        'mental_wellness_index_0_100'
    ]
    for col in numeric_cols_with_nan:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    gender_map = {'Male': 0, 'Female': 1}
    df['gender'] = df['gender'].map(gender_map)
    if df['gender'].isnull().any():
        mode_gender = df['gender'].mode()[0]
        df['gender'] = df['gender'].fillna(mode_gender)

    df = pd.get_dummies(
        df, columns=['occupation', 'work_mode'], drop_first=True
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(
        'user_id', errors='ignore'
    )
    df_num = df[numeric_cols].dropna(how="all")
    z_scores = np.abs(stats.zscore(df_num, nan_policy='omit'))
    mask = (z_scores < 3).all(axis=1)
    df_clean = df.loc[df_num.index[mask]]
    
    logging.info("Предобработка завершена.")
    return df_clean


def classification_training(data: pd.DataFrame):
    """
    Выполняет обучение и оценку моделей KNN и Decision Tree на
    предобработанных данных, выводит метрики и строит дерево решений.
    """
    
    data = prepare_data(data)
    
    if not isinstance(data, pd.DataFrame):
        logging.error(f"Ожидается DataFrame, получен {type(data)}.")
        return

    df = data.copy()

    df['mental_wellness_index_0_100'] = (
        df['mental_wellness_index_0_100'] >= 15
    ).astype(int)

    X = df.drop(columns=['user_id', 'mental_wellness_index_0_100'])
    y = df['mental_wellness_index_0_100']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обоснование выбора гиперпараметров:
    # KNN (n_neighbors=5): стандартное значение, обеспечивающее баланс
    # между гибкостью модели и устойчивостью к шуму.
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # DecisionTree (max_depth=4, min_samples_leaf=10): параметры
    # ограничивают сложность дерева, предотвращая переобучение и делая
    # модель более робастной. random_state=42 для воспроизводимости.
    dt = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=10, random_state=42
    )

    models = {
        'KNN': knn,
        'DT': dt
    }

    for model_name, model in models.items():
        if model_name == 'KNN':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Вывод метрик в консоль строго по формату ТЗ.
        print(f"{model_name}: {accuracy}; {f1}")

    logging.info("Построение дерева решений...")
    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        dt,
        feature_names=X.columns.tolist(),
        class_names=['<15', '>=15'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Визуализация дерева решений (Decision Tree)")
    plt.show()


def main():
    """
    Главная функция для выполнения всего пайплайна.
    """
    yandex_disk_url = "https://disk.yandex.ru/d/NVmydyRuUzCxPw"
    file_content_bytes = download_yandex_disk_file(yandex_disk_url)

    if file_content_bytes:
        try:
            raw_data = pd.read_csv(
                BytesIO(file_content_bytes),
                sep='\t',
                engine='python',
                on_bad_lines='skip'
            )
            classification_training(data=raw_data)

        except Exception as e:
            logging.error(f"Произошла ошибка при обработке данных: {e}")


if __name__ == "__main__":
    main()