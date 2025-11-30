# Road Lane Marking Segmentation

Проект для семантической сегментации дорожной разметки с использованием DeepLabV3+ и EfficientNet-b4.

## 🚀 Быстрый старт

### Хотите обучить на GPU в облаке? (РЕКОМЕНДУЕТСЯ)
👉 **Читайте [GPU_SETUP.md](GPU_SETUP.md)** - пошаговая инструкция для Google Colab

### Хотите обучить локально на CPU?
```bash
python train.py
```
⚠️ Обучение будет медленнее, но работает

## Описание

Данный проект реализует обучение нейронной сети для сегментации 25 классов дорожной разметки в различных погодных и световых условиях.

### Особенности

- **Архитектура**: DeepLabV3+ с EfficientNet-b4 encoder
- **Transfer Learning**: Предобученные веса ImageNet
- **Аугментации**: Albumentations для расширения датасета
- **Mixed Precision**: FP16 для оптимизации обучения
- **Визуализация**: TensorBoard для отслеживания метрик
- **Метрики**: IoU, Dice, Pixel Accuracy

### Классы разметки (25 классов)

1. background
2. box junction
3. crosswalk
4. stop line
5. solid single white
6. solid single yellow
7. solid single red
8. solid double white
9. solid double yellow
10. dashed single white
11. dashed single yellow
12. left arrow
13. straight arrow
14. right arrow
15. left straight arrow
16. right straight arrow
17. channelizing line
18. motor prohibited
19. slow
20. motor priority lane
21. motor waiting zone
22. left turn box
23. motor icon
24. bike icon
25. parking lot

## Структура проекта

```
diplom/
├── dataset/                    # Директория с данными
│   ├── images-*/              # Изображения
│   ├── labels-*/              # Маски сегментации
│   ├── clear-*/               # Чистая погода
│   ├── night-*/               # Ночные условия
│   ├── rainy-*/               # Дождь
│   └── rlmd.csv               # Карта классов
├── config.py                  # Конфигурация
├── dataset.py                 # Загрузка данных и аугментации
├── model.py                   # Архитектура модели
├── train.py                   # Скрипт обучения
├── inference.py               # Скрипт предсказаний
├── utils.py                   # Утилиты и метрики
├── requirements.txt           # Зависимости
└── README.md                  # Документация
```

## Установка

### 1. Клонирование репозитория

```bash
cd /home/tsokurenkosv/projects/diplom
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### Основные зависимости:

- torch >= 2.0.0
- torchvision >= 0.15.0
- segmentation-models-pytorch >= 0.3.3
- albumentations >= 1.3.1
- opencv-python >= 4.8.0
- tensorboard >= 2.13.0

## Использование

### Обучение модели

Базовое обучение с параметрами по умолчанию:

```bash
python train.py
```

### Настройка параметров

Вы можете изменить параметры в `config.py`:

```python
config = Config(
    batch_size=8,           # Размер батча
    num_epochs=50,          # Количество эпох
    learning_rate=5e-4,     # Learning rate
    image_size=512,         # Размер изображений
    encoder_name='efficientnet-b4'
)
```

### Inference (предсказания)

Для одного изображения:

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --output_dir predictions
```

Для пакета изображений:

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image_dir path/to/images/ \
    --output_dir predictions \
    --visualize
```

### Мониторинг обучения

Запустите TensorBoard для просмотра метрик:

```bash
tensorboard --logdir runs
```

Откройте браузер: http://localhost:6006

## Конфигурация

### Основные параметры в `config.py`:

| Параметр | Значение | Описание |
|----------|----------|----------|
| `batch_size` | 4 | Размер батча |
| `num_epochs` | 100 | Количество эпох |
| `learning_rate` | 1e-4 | Learning rate |
| `image_size` | 512 | Размер изображения |
| `encoder_name` | efficientnet-b4 | Encoder модели |
| `use_amp` | True | Mixed Precision |
| `early_stopping_patience` | 15 | Ранняя остановка |

### Аугментации

В `dataset.py` используются следующие аугментации:

- HorizontalFlip
- RandomBrightnessContrast
- GaussNoise
- GaussianBlur
- ColorJitter
- ShiftScaleRotate

## Метрики

### Отслеживаемые метрики:

1. **mIoU (mean Intersection over Union)** - основная метрика
2. **Dice coefficient** - коэффициент Dice
3. **Pixel Accuracy** - точность на уровне пикселей
4. **Loss** - функция потерь

## Результаты

После обучения модель сохраняется в `checkpoints/`:

- `best_model.pth` - лучшая модель по mIoU
- `checkpoint_epoch_N.pth` - чекпоинты каждые N эпох

Логи TensorBoard сохраняются в `runs/`.

## Примеры использования

### Изменение архитектуры encoder

```python
# В config.py
config = Config(
    encoder_name='resnet50',  # или 'mobilenet_v2', 'resnet101'
    encoder_weights='imagenet'
)
```

### Использование разных learning rates

```python
# В config.py
config = Config(
    use_diff_lr=True,
    learning_rate=1e-3,
    encoder_lr_factor=0.1  # encoder LR = 1e-4
)
```

### Отключение Mixed Precision

```python
# В config.py
config = Config(
    use_amp=False
)
```

## Требования к системе

### Для обучения на GPU:
- NVIDIA GPU с минимум 6GB VRAM (рекомендуется 8GB+)
- CUDA 11.0+
- Python 3.8+

### Для обучения на CPU (текущая конфигурация):
- Python 3.8+
- 8GB+ RAM
- ⚠️ Обучение будет значительно медленнее

## Советы по оптимизации

1. **Уменьшите batch_size** если не хватает памяти GPU
2. **Используйте gradient accumulation** для симуляции большего batch size
3. **Попробуйте разные encoder'ы** для баланса скорость/точность:
   - MobileNetV2 - быстрый, легкий
   - ResNet50 - баланс
   - EfficientNet-b4 - точный, но медленнее

4. **Настройте аугментации** под ваш конкретный случай

## Troubleshooting

### CUDA Out of Memory

```python
# Уменьшите batch_size в config.py
config = Config(batch_size=2)

# Или уменьшите image_size
config = Config(image_size=384)
```

### Медленное обучение

```python
# Увеличьте num_workers (на GPU)
config = Config(num_workers=8)

# Используйте более легкий encoder
config = Config(encoder_name='mobilenet_v2')

# Уменьшите размер изображения
config = Config(image_size=384)
```

## Использование GPU

### Проверка доступности GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Если у вас нет NVIDIA GPU

#### Вариант 1: AMD GPU (ROCm)
Если у вас AMD GPU, установите PyTorch с ROCm:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

⚠️ **Примечание**: Не все модели AMD GPU поддерживают ROCm. Проверьте совместимость.

#### Вариант 2: Использовать облачные сервисы (рекомендуется)

1. **Google Colab** (бесплатно): https://colab.research.google.com
   - Бесплатный NVIDIA T4 GPU
   - До 12 часов непрерывной работы

2. **Kaggle Notebooks** (бесплатно): https://www.kaggle.com/code
   - Бесплатный GPU (P100 или T4)
   - До 30 часов в неделю

3. **Paperspace Gradient** (бесплатный tier): https://www.paperspace.com/gradient

##### Пример использования в Google Colab:

```python
# 1. Загрузите проект в Colab
!git clone your-repo-url
%cd diplom

# 2. Установите зависимости
!pip install -r requirements.txt

# 3. Загрузите датасет (используйте Google Drive)
from google.colab import drive
drive.mount('/content/drive')

# 4. Обновите путь к датасету в config.py
config = Config(
    data_dir='/content/drive/MyDrive/dataset',
    batch_size=16,  # Можно увеличить на GPU
    num_workers=2,
    image_size=512
)

# 5. Запустите обучение
!python train.py
```

#### Вариант 3: Обучение на CPU (текущая конфигурация)

Код уже оптимизирован для CPU. Настройки:
- `batch_size=2` (уменьшено для CPU)
- `num_workers=2` (уменьшено для CPU)
- `image_size=512` (можно уменьшить до 256 или 384 для ускорения)

Для ускорения на CPU:
```python
# В config.py
config = Config(
    batch_size=2,
    image_size=384,  # Уменьшите размер для ускорения
    encoder_name='mobilenet_v2',  # Более легкая модель
    num_workers=2
)
```
