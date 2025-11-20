# Структура проекта MFF-GBDT

## Трехэтапный Pipeline

### 1. Segmentation (Сегментация) - `segmentation/`
**Назначение**: Обучение и использование Mask R-CNN для сегментации бройлеров на depth-изображениях.

```
segmentation/
├── training/
│   ├── train.py              # Обучение Mask R-CNN
│   └── __init__.py
├── inference/
│   ├── test.py               # Тестирование/инференс Mask R-CNN
│   └── __init__.py
├── models/
│   ├── Mask_rcnn_Model.py    # Архитектура Mask R-CNN
│   └── __init__.py
└── datasets/
    ├── Penn_Fudan_dataset.py # Dataset для обучения сегментации
    └── __init__.py
```

### 2. Features (Извлечение признаков) - `features/`
**Назначение**: Извлечение 25 hand-crafted 2D/3D признаков + 2048 ResNet50 признаков.

```
features/
├── extraction/
│   ├── manual_features.py    # Извлечение 25 hand-crafted признаков
│   ├── resnet_features.py   # Извлечение ResNet50 признаков
│   └── __init__.py
├── training/
│   ├── train_resnet.py       # Обучение ResNet50 для извлечения признаков
│   ├── train_fusion.py      # Обучение FusionNet (экспериментально)
│   ├── train_fc.py          # Обучение FC (экспериментально)
│   └── __init__.py
├── models/
│   ├── FusonNet.py          # Архитектура FusionNet
│   ├── FC.py                 # Архитектура FC
│   └── __init__.py
├── datasets/
│   ├── chicken200.py         # Dataset для обучения ResNet
│   └── __init__.py
└── testing/
    ├── test_resnet.py        # Тестирование ResNet
    ├── test_fusion.py        # Тестирование FusionNet
    └── test_fc.py            # Тестирование FC
```

### 3. GBDT (Оценка веса) - `gbdt/`
**Назначение**: Обучение и использование LightGBM/XGBoost для предсказания веса по признакам.

```
gbdt/
├── training/
│   ├── train_lightgbm.py    # Обучение LightGBM (основной)
│   ├── train_xgboost.py     # Обучение XGBoost (экспериментально)
│   └── __init__.py
├── inference/
│   ├── predict.py            # Предсказание веса по признакам
│   └── __init__.py
└── data/
    └── csvData/              # CSV с признаками (существующая структура)
```

### 4. Inference (MVP Pipeline) - `inference/`
**Назначение**: End-to-end pipeline для инференса (будущие файлы).

```
inference/
├── segmentation.py          # Загрузка и использование Mask R-CNN
├── feature_extract.py       # Извлечение всех признаков
├── predict_weight.py        # Предсказание веса через GBDT
├── launch.py                # Главный entrypoint
└── __init__.py
```

### 5. Utils (Утилиты) - `utils/`
**Назначение**: Общие утилиты для всех этапов.

```
utils/
├── engine.py                # Training/evaluation engine
├── transforms.py            # Трансформации изображений
├── coco_utils.py            # COCO utilities
├── coco_eval.py             # COCO evaluation
└── __init__.py
```

### 6. Data (Данные) - `data/`
**Назначение**: Хранение данных и результатов.

```
data/
├── raw/                     # Исходные данные
│   ├── coco_sets/           # Данные для обучения Mask R-CNN
│   └── ...
├── processed/               # Обработанные данные
│   ├── csvData/             # CSV с признаками
│   └── ...
├── models/                  # Обученные модели
│   ├── segmentation/        # Mask R-CNN веса
│   ├── features/            # ResNet веса
│   └── gbdt/                # LightGBM/XGBoost модели
└── outputs/                 # Результаты экспериментов
    ├── exps/                # Существующие эксперименты
    └── ...
```

### 7. Config (Конфигурация) - корень проекта
```
config.yaml                  # Конфигурация для всех этапов
requirements.txt             # Зависимости
README.md                    # Документация
```

## Текущее состояние → Новая структура

### Файлы для перемещения:

**Segmentation:**
- `train.py` → `segmentation/training/train.py`
- `test.py` → `segmentation/inference/test.py`
- `Mask_rcnn_Model.py` → `segmentation/models/Mask_rcnn_Model.py`
- `Penn_Fudan_dataset.py` → `segmentation/datasets/Penn_Fudan_dataset.py`

**Features:**
- `GBDT/dataset/20210206-1198.py` → `features/extraction/manual_features.py`
- `GBDT/dataset/featureExtraction.py` → `features/extraction/resnet_features.py`
- `model/trainResnet.py` → `features/training/train_resnet.py`
- `model/trainFuson.py` → `features/training/train_fusion.py`
- `model/trainFC.py` → `features/training/train_fc.py`
- `model/FusonNet.py` → `features/models/FusonNet.py`
- `model/FC.py` → `features/models/FC.py`
- `model/testResnet.py` → `features/testing/test_resnet.py`
- `model/testFusonnet.py` → `features/testing/test_fusion.py`
- `model/testFC.py` → `features/testing/test_fc.py`
- `dataset/chicken200/chicken200.py` → `features/datasets/chicken200.py`

**GBDT:**
- `GBDT/predWeight/trainLGBM.py` → `gbdt/training/train_lightgbm.py`
- `GBDT/predWeight/trainXGB.py` → `gbdt/training/train_xgboost.py`
- `GBDT/predWeight/predict.py` → `gbdt/inference/predict.py`

**Utils:**
- `engine.py` → `utils/engine.py`
- `utils.py` → `utils/general.py` (переименовать для избежания конфликта)
- `transforms.py` → `utils/transforms.py`
- `coco_utils.py` → `utils/coco_utils.py`
- `coco_eval.py` → `utils/coco_eval.py`
- `util/xml2mask.py` → `utils/xml2mask.py`

**Data:**
- `GBDT/csvData/` → `data/processed/csvData/`
- `weight/` → `data/models/segmentation/`
- `exps/` → `data/outputs/exps/`
- `coco_sets/` → `data/raw/coco_sets/`

