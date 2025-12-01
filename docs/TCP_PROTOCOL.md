# TCP Protocol Specification для приборной панели

## Общая структура кадра (Frame)

Все сообщения имеют единую структуру:

```
[SYNC] [HEADER] [PAYLOAD] [CRC16]
```

### Структура заголовка

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| SYNC_BYTE | uint8 | 1 byte | Синхробайт = 0xAA |
| VERSION | uint8 | 1 byte | Версия протокола = 0x01 |
| MSG_TYPE | uint8 | 1 byte | Тип сообщения (0x01-0x05) |
| SEQ | uint8 | 1 byte | Порядковый номер кадра (0-255, циклический) |
| TIMESTAMP | uint32 | 4 bytes | Метка времени в миллисекундах (монотонное время) |
| PAYLOAD_LEN | uint16 | 2 bytes | Длина payload в байтах |
| **Всего** | | **9 bytes** | |
| PAYLOAD | bytes | variable | Данные (зависит от типа сообщения) |
| CRC16 | uint16 | 2 bytes | CRC16-IBM (0xA001) от [VERSION..PAYLOAD] |

**Общий размер кадра:** `1 + 9 + PAYLOAD_LEN + 2` bytes

---

## Типы сообщений (MSG_TYPE)

### 0x01 - LANE_SUMMARY (Краткая информация о полосе)

**Размер payload:** 6 bytes
**Общий размер кадра:** 18 bytes

Содержит базовую информацию о положении полосы движения.

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| left_offset_dm | int16 | 2 bytes | Смещение левой границы в дециметрах |
| right_offset_dm | int16 | 2 bytes | Смещение правой границы в дециметрах |
| left_type | uint8 | 1 byte | Тип левой линии (0=unknown, 1=solid, 2=double, 3=dashed) |
| right_type | uint8 | 1 byte | Тип правой линии |
| allowed_maneuvers | uint8 | 1 byte | Разрешенные маневры (битовая маска) |
| quality | uint8 | 1 byte | Качество детектирования (0-255) |

**Пример интерпретации:**
```python
left_offset_dm = -15  # 1.5 метра влево
right_offset_dm = 25  # 2.5 метра вправо
# Ширина полосы = 4.0 метра
```

---

### 0x02 - MARKING_OBJECTS (Объекты разметки - базовая версия)

**Размер payload:** `1 + N * 13` bytes (N ≤ 78 объектов)
**Общий размер кадра:** `12 + N * 13` bytes

Содержит информацию о дорожных объектах (стрелки, переходы, стоп-линии и т.д.)

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| count | uint8 | 1 byte | Количество объектов N |

**Для каждого объекта (13 bytes):**

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| class_id | uint8 | 1 byte | ID класса объекта (см. таблицу классов) |
| x_dm | int16 | 2 bytes | Позиция X в дециметрах (относительно камеры) |
| y_dm | int16 | 2 bytes | Позиция Y в дециметрах (вперёд от камеры) |
| length_dm | uint16 | 2 bytes | Длина объекта в дециметрах |
| width_dm | uint16 | 2 bytes | Ширина объекта в дециметрах |
| yaw_decideg | int16 | 2 bytes | Ориентация в децидеградусах (0.1°) |
| confidence | uint8 | 1 byte | Уверенность детектирования (0-255) |
| flags | uint8 | 1 byte | Флаги состояния (битовая маска) |

---

### 0x03 - LANE_DETAILS (Детальная информация о границах полосы)

**Размер payload:** 32 bytes
**Общий размер кадра:** 44 bytes

Расширенная информация о границах полосы с точками траектории.

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| left_type | uint8 | 1 byte | Тип левой линии |
| right_type | uint8 | 1 byte | Тип правой линии |
| left_color | uint8 | 1 byte | Цвет левой линии (0=white, 1=yellow, 2=red) |
| right_color | uint8 | 1 byte | Цвет правой линии |
| left_quality | uint8 | 1 byte | Качество левой границы (0-255) |
| right_quality | uint8 | 1 byte | Качество правой границы (0-255) |
| left_width_dm | uint16 | 2 bytes | Ширина левой линии в дециметрах |
| right_width_dm | uint16 | 2 bytes | Ширина правой линии в дециметрах |

**Далее 6 точек траектории (по 3 для каждой стороны):**

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| left_point[0..2].x_dm | int16 | 2 bytes × 3 | X координаты левой границы |
| left_point[0..2].y_dm | int16 | 2 bytes × 3 | Y координаты левой границы |
| right_point[0..2].x_dm | int16 | 2 bytes × 3 | X координаты правой границы |
| right_point[0..2].y_dm | int16 | 2 bytes × 3 | Y координаты правой границы |

**Всего:** 8 + 24 = 32 bytes

---

### 0x04 - MARKING_OBJECTS_EX (Объекты разметки - расширенная версия)

**Размер payload:** `1 + N * 15` bytes (N ≤ 68 объектов)
**Общий размер кадра:** `12 + N * 15` bytes

Расширенная версия объектов с информацией о цвете и стиле линий.

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| count | uint8 | 1 byte | Количество объектов N |

**Для каждого объекта (15 bytes):**

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| class_id | uint8 | 1 byte | ID класса объекта |
| x_dm | int16 | 2 bytes | Позиция X в дециметрах |
| y_dm | int16 | 2 bytes | Позиция Y в дециметрах |
| length_dm | uint16 | 2 bytes | Длина объекта в дециметрах |
| width_dm | uint16 | 2 bytes | Ширина объекта в дециметрах |
| yaw_decideg | int16 | 2 bytes | Ориентация в децидеградусах |
| confidence | uint8 | 1 byte | Уверенность детектирования (0-255) |
| flags | uint8 | 1 byte | Флаги состояния |
| line_color | uint8 | 1 byte | Цвет линии (0=white, 1=yellow, 2=red) |
| line_style | uint8 | 1 byte | Стиль линии (0=unknown, 1=solid, 2=double, 3=dashed) |

---

### 0x05 - FITTED_LINES (Полиномиальные кривые линий) ⭐ НОВОЕ

**Размер payload:** `1 + N * 23` bytes (N ≤ 44 линий)
**Общий размер кадра:** `12 + N * 23` bytes

Содержит полиномиальные аппроксимации линий дорожной разметки для 3D реконструкции.

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| count | uint8 | 1 byte | Количество линий N |

**Для каждой линии (23 bytes):**

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| class_id | uint8 | 1 byte | Тип линии (4-10, см. таблицу классов) |
| side | uint8 | 1 byte | Сторона: 0=unknown, 1=left, 2=right, 3=center |
| color | uint8 | 1 byte | Цвет линии: 0=white, 1=yellow, 2=red |
| style | uint8 | 1 byte | Стиль: 0=unknown, 1=solid, 2=double, 3=dashed |
| poly_a | float32 | 4 bytes | Коэффициент 'a' полинома x = ay² + by + c |
| poly_b | float32 | 4 bytes | Коэффициент 'b' полинома |
| poly_c | float32 | 4 bytes | Коэффициент 'c' полинома |
| y_start | int16 | 2 bytes | Начальная Y координата в пикселях |
| y_end | int16 | 2 bytes | Конечная Y координата в пикселях |
| confidence | uint8 | 1 byte | Уверенность детектирования (0-255) |
| quality | uint8 | 1 byte | Качество фитирования - inlier ratio (0-255) |

**Формула реконструкции линии:**
```
x = a * y² + b * y + c
где y ∈ [y_start, y_end]
```

**Пример использования:**
```python
# Получили линию с коэффициентами:
a, b, c = 0.0005, -0.3, 250.0
y_start, y_end = 50, 350

# Генерируем точки линии
y_values = np.linspace(y_start, y_end, 100)
x_values = a * y_values**2 + b * y_values + c

# Теперь (x_values, y_values) - это координаты линии в пиксельном пространстве
# Для 3D нужно спроецировать через калибровку камеры
```

---

## Таблица классов объектов (class_id)

| class_id | Название | Описание |
|----------|----------|----------|
| 1 | box_junction | Перекрёсток с сеткой |
| 2 | crosswalk | Пешеходный переход |
| 3 | stop_line | Стоп-линия |
| 4 | solid_single_white | Одинарная сплошная белая |
| 5 | solid_single_yellow | Одинарная сплошная жёлтая |
| 6 | solid_single_red | Одинарная сплошная красная |
| 7 | double_white | Двойная белая |
| 8 | double_yellow | Двойная жёлтая |
| 9 | dashed_white | Пунктирная белая |
| 10 | dashed_yellow | Пунктирная жёлтая |
| 11 | arrow_left | Стрелка влево |
| 12 | arrow_straight | Стрелка прямо |
| 13 | arrow_right | Стрелка вправо |
| 14 | arrow_left_straight | Стрелка влево+прямо |
| 15 | arrow_right_straight | Стрелка вправо+прямо |
| 16 | channelizing_line | Направляющая линия |
| 22 | motor_icon | Иконка автомобиля |
| 23 | bike_icon | Иконка велосипеда |

---

## Последовательность сообщений

При каждом обновлении детектирования отправляется **5 сообщений** в следующем порядке:

1. **LANE_SUMMARY** (0x01) - базовая информация о полосе
2. **MARKING_OBJECTS** (0x02) - объекты разметки (базовая)
3. **LANE_DETAILS** (0x03) - детальные границы полосы
4. **MARKING_OBJECTS_EX** (0x04) - объекты разметки (расширенная)
5. **FITTED_LINES** (0x05) - ⭐ полиномиальные кривые линий

**Частота:** ~10-30 Hz (зависит от скорости обработки)

---

## Система координат

### 2D координаты (пиксели)
- **Начало координат:** Верхний левый угол изображения
- **X ось:** Вправо (0..width)
- **Y ось:** Вниз (0..height)

### 3D координаты (дециметры)
- **Начало координат:** Позиция камеры
- **X ось:** Вправо (отрицательные значения = влево)
- **Y ось:** Вперёд по направлению движения (всегда положительные)
- **Z ось:** Вверх (не используется в текущей версии)

**Единица измерения:** Дециметры (dm)
- 1 dm = 10 см = 0.1 м
- 10 dm = 1 м

---

## Реконструкция 3D сцены на приборной панели

### Шаг 1: Получение данных

Приборная панель получает 5 типов сообщений и сохраняет последние данные:
- `LaneSummary` - положение полосы
- `MarkingObjects` - объекты на дороге
- `LaneDetails` - точки границ
- `FittedLines` - полиномиальные кривые ⭐

### Шаг 2: Преобразование координат

#### Для FITTED_LINES (полиномов):
```python
# 1. Генерация точек линии в 2D (пиксели)
y_values = np.linspace(line.y_start, line.y_end, 50)
x_values = line.poly_a * y_values**2 + line.poly_b * y_values + line.poly_c

# 2. Преобразование из пикселей в 3D метры (через калибровку камеры)
for x_px, y_px in zip(x_values, y_values):
    # Используем калибровочную матрицу камеры
    X_3d, Y_3d, Z_3d = pixel_to_world(x_px, y_px, camera_matrix, height_above_road)

    # X_3d - поперечное смещение в метрах
    # Y_3d - расстояние вперёд в метрах
    # Z_3d - высота (обычно 0 для разметки на дороге)
```

#### Для MARKING_OBJECTS:
```python
# Позиция уже в дециметрах (dm)
x_meters = obj.x_dm / 10.0  # Конвертация в метры
y_meters = obj.y_dm / 10.0
length_m = obj.length_dm / 10.0
width_m = obj.width_dm / 10.0
yaw_deg = obj.yaw_decideg / 10.0  # Ориентация в градусах
```

### Шаг 3: Отрисовка в 3D

```python
# Для каждой FITTED_LINE:
if line.style == SOLID:
    draw_solid_line_3d(points_3d, color=line.color, width=0.1)
elif line.style == DASHED:
    draw_dashed_line_3d(points_3d, color=line.color, width=0.1, dash_length=3.0)
elif line.style == DOUBLE:
    draw_double_line_3d(points_3d, color=line.color, offset=0.15)

# Для каждого MARKING_OBJECT:
if obj.class_id == ARROW_STRAIGHT:
    draw_arrow_3d(position=(x, y, 0), yaw=yaw_deg, length=length_m)
elif obj.class_id == CROSSWALK:
    draw_crosswalk_3d(position=(x, y, 0), size=(length_m, width_m), yaw=yaw_deg)
```

### Шаг 4: Камера и перспектива

```python
# Настройка камеры для вида "из автомобиля"
camera_position = (0, 0, 1.2)  # 1.2м над дорогой
camera_look_at = (0, 50, 0)     # Смотрим вперёд на 50м
camera_up = (0, 0, 1)           # Z - вверх

# Отрисовка дороги
draw_road_surface(width=6.0, length=100.0, color=gray)

# Отрисовка автомобиля (для контекста)
draw_ego_vehicle(position=(0, 0, 0), model=car_3d_model)
```

---

## Пример полного парсинга на приборной панели

```python
import struct

def parse_frame(data: bytes):
    # Проверка синхробайта
    if data[0] != 0xAA:
        return None

    # Парсинг заголовка
    version, msg_type, seq, timestamp, payload_len = struct.unpack("<BBBIH", data[1:10])

    # Извлечение payload
    payload = data[10:10+payload_len]

    # Проверка CRC
    crc_received = struct.unpack("<H", data[10+payload_len:12+payload_len])[0]
    crc_calculated = crc16_ibm(data[1:10+payload_len])
    if crc_received != crc_calculated:
        return None

    # Парсинг в зависимости от типа
    if msg_type == 0x05:  # FITTED_LINES
        return parse_fitted_lines(payload)
    elif msg_type == 0x04:  # MARKING_OBJECTS_EX
        return parse_marking_objects_ex(payload)
    # ... остальные типы

def parse_fitted_lines(payload: bytes):
    count = struct.unpack("<B", payload[0:1])[0]
    lines = []

    offset = 1
    for i in range(count):
        line_data = struct.unpack("<BBBBfffhhBB", payload[offset:offset+23])
        line = {
            'class_id': line_data[0],
            'side': line_data[1],  # 0=unknown, 1=left, 2=right, 3=center
            'color': line_data[2],
            'style': line_data[3],
            'poly_a': line_data[4],
            'poly_b': line_data[5],
            'poly_c': line_data[6],
            'y_start': line_data[7],
            'y_end': line_data[8],
            'confidence': line_data[9],
            'quality': line_data[10],
        }
        lines.append(line)
        offset += 23

    return lines
```

---

## Преимущества нового сообщения FITTED_LINES (0x05)

✅ **Точная геометрия:** Полиномиальная аппроксимация точно описывает изогнутые линии
✅ **Компактность:** Всего 23 байта вместо отправки сотен точек
✅ **Гладкость:** Можно генерировать любое количество точек для плавной отрисовки
✅ **Метаданные:** Содержит цвет, стиль, сторону, качество
✅ **3D готовность:** Легко конвертируется в 3D координаты через калибровку

---

## Частые вопросы (FAQ)

**Q: Почему полином x = f(y), а не y = f(x)?**
A: Потому что линии разметки почти вертикальные в изображении. Полином y = f(x) не может описать вертикальные линии (разрыв функции).

**Q: Как часто приходят сообщения?**
A: ~10-30 Hz, зависит от производительности. Каждый кадр генерирует 5 сообщений подряд.

**Q: Нужно ли буферизовать сообщения?**
A: Рекомендуется хранить только последнее значение каждого типа для минимальной задержки.

**Q: Что делать с перекрывающимися объектами?**
A: Система уже фильтрует их по приоритету. Стрелки имеют высший приоритет.

**Q: Как калибровать камеру?**
A: Используйте стандартные методы калибровки OpenCV с шахматной доской. Матрица камеры нужна для проекции 2D→3D.

---

## История изменений

- **v1.0** - Базовые сообщения (0x01-0x04)
- **v1.1** - Добавлено сообщение FITTED_LINES (0x05) для 3D реконструкции
