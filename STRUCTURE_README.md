# Структура проекта MFF-GBDT

## Обзор

Проект организован по **трехэтапному pipeline** для оценки веса бройлеров:

1. **Segmentation** → 2. **Features** → 3. **GBDT**

---

## Структура директорий

```
MFF-GBDT/
├── segmentation/          # Этап 1: Сегментация (Mask R-CNN)
│   ├── training/          # Обучение Mask R-CNN
│   ├── inference/         # Инференс Mask R-CNN
│   ├── models/            # Архитектура модели
│   └── datasets/          # Dataset для обучения
│
├── features/              # Этап 2: Извлечение признаков
│   ├── extraction/        # Скрипты извлечения признаков
│   ├── training/          # Обучение ResNet/FusionNet
│   ├── models/            # Архитектуры моделей
│   ├── datasets/          # Dataset для обучения
│   └── testing/           # Тестирование моделей
│
├── gbdt/                  # Этап 3: GBDT (LightGBM/XGBoost)
│   ├── training/          # Обучение GBDT
│   └── inference/         # Инференс GBDT
│
├── inference/             # MVP: End-to-end pipeline (будущие файлы)
│
├── utils/                 # Общие утилиты
│
└── data/                  # Данные и результаты
    ├── raw/               # Исходные данные
    ├── processed/         # Обработанные данные
    ├── models/            # Обученные модели
    └── outputs/           # Результаты экспериментов
```

---

## Детальное описание модулей

### 1. Segmentation (`segmentation/`)

**Назначение**: Обучение и использование Mask R-CNN для сегментации бройлеров на depth-изображениях.

**Файлы**:
- `training/train.py` - Обучение Mask R-CNN
- `inference/test.py` - Тестирование/инференс Mask R-CNN
- `models/Mask_rcnn_Model.py` - Архитектура Mask R-CNN
- `datasets/Penn_Fudan_dataset.py` - Dataset для обучения сегментации

**Входные данные**: `data/raw/coco_sets/mixData/`
**Выходные данные**: Маски и обрезанные изображения бройлеров
**Модели**: `data/models/segmentation/weight/`

---

### 2. Features (`features/`)

**Назначение**: Извлечение 25 hand-crafted 2D/3D признаков + 2048 ResNet50 признаков.

**Файлы**:
- `extraction/manual_features.py` - Извлечение 25 hand-crafted признаков
- `extraction/resnet_features.py` - Извлечение ResNet50 признаков
- `training/train_resnet.py` - Обучение ResNet50 для извлечения признаков
- `training/train_fusion.py` - Обучение FusionNet (экспериментально)
- `training/train_fc.py` - Обучение FC (экспериментально)
- `models/FusonNet.py` - Архитектура FusionNet
- `models/FC.py` - Архитектура FC
- `datasets/chicken200.py` - Dataset для обучения ResNet
- `testing/` - Тестирование моделей

**Входные данные**: Маски и обрезанные изображения из этапа Segmentation
**Выходные данные**: CSV с признаками в `data/processed/csvData/`
**Модели**: `data/outputs/exps/myresnet/`

---

### 3. GBDT (`gbdt/`)

**Назначение**: Обучение и использование LightGBM/XGBoost для предсказания веса по признакам.

**Файлы**:
- `training/train_lightgbm.py` - Обучение LightGBM (основной)
- `training/train_xgboost.py` - Обучение XGBoost (экспериментально)
- `inference/predict.py` - Предсказание веса по признакам

**Входные данные**: CSV с признаками из `data/processed/csvData/`
**Выходные данные**: Предсказания веса
**Модели**: `data/outputs/exps/lgbm_data_*/` или `xgb_data_*/`

---

### 4. Utils (`utils/`)

**Общие утилиты для всех этапов**:
- `engine.py` - Training/evaluation engine
- `general.py` - Общие функции (бывший `utils.py`)
- `transforms.py` - Трансформации изображений
- `coco_utils.py` - COCO utilities
- `coco_eval.py` - COCO evaluation
- `xml2mask.py` - Конвертация XML в маски

---

### 5. Data (`data/`)

**Организация данных**:
- `raw/` - Исходные данные (например, `coco_sets/`)
- `processed/` - Обработанные данные:
  - `csvData/` - CSV с признаками
  - `dataset/` - Train/test splits
- `models/` - Обученные модели:
  - `segmentation/` - Mask R-CNN веса
  - `features/` - ResNet веса
  - `gbdt/` - LightGBM/XGBoost модели
- `outputs/exps/` - Результаты экспериментов

---

## Типичные пути

### Обучение
```bash
# Segmentation
python segmentation/training/train.py

# Features (ResNet)
python features/training/train_resnet.py

# GBDT (LightGBM)
python gbdt/training/train_lightgbm.py
```

### Инференс
```bash
# Segmentation
python segmentation/inference/test.py

# GBDT
python gbdt/inference/predict.py
```

---

## Важные замечания

1. **Импорты**: Все файлы используют относительные импорты с добавлением `project_root` в `sys.path`
2. **Пути**: Все пути к данным и моделям обновлены на новую структуру (`data/raw/`, `data/processed/`, `data/models/`, `data/outputs/`)
3. **Этапы**: Pipeline строго разделен на три этапа - не смешивайте логику между этапами
4. **MVP**: Директория `inference/` предназначена для будущих файлов end-to-end pipeline

---

## Миграция

Если вы используете старые пути, обновите их:
- `GBDT/csvData/` → `data/processed/csvData/`
- `exps/` → `data/outputs/exps/`
- `weight/` → `data/models/segmentation/weight/`
- `coco_sets/` → `data/raw/coco_sets/`
- `dataset/train|test/` → `data/processed/dataset/train|test/`

