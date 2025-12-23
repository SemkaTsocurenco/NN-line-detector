# TCP Protocol V2 Specification для приборной панели

## Обзор изменений V2

Протокол версии 2 упрощает структуру сообщений с **5 типов до 2 типов**:
- **0x01 - LANE_LINES**: Линии дорожной разметки с полиномами и точками в метрах
- **0x02 - ROAD_OBJECTS**: Объекты на дороге (стрелки, переходы, стоп-линии)

**Ключевые улучшения:**
✅ Все координаты в метрах (используется перспективное преобразование)
✅ Система координат: Y вперёд, X вправо, начало в камере
✅ Упрощенная структура - только 2 типа сообщений вместо 5
✅ Отправляются только те данные, которые рисуются на экране

---

## Общая структура кадра (Frame)

Все сообщения имеют единую структуру:

```
[SYNC] [HEADER] [PAYLOAD] [CRC16]
```

### Структура заголовка

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| SYNC_BYTE | uint8 | 1 byte | Синхробайт = 0xAA |
| VERSION | uint8 | 1 byte | Версия протокола = 0x02 |
| MSG_TYPE | uint8 | 1 byte | Тип сообщения (0x01-0x02) |
| SEQ | uint8 | 1 byte | Порядковый номер кадра (0-255, циклический) |
| TIMESTAMP | uint32 | 4 bytes | Метка времени в миллисекундах (монотонное время) |
| PAYLOAD_LEN | uint16 | 2 bytes | Длина payload в байтах |
| **Всего** | | **9 bytes** | |
| PAYLOAD | bytes | variable | Данные (зависит от типа сообщения) |
| CRC16 | uint16 | 2 bytes | CRC16-IBM (0xA001) от [VERSION..PAYLOAD] |

**Общий размер кадра:** `1 + 9 + PAYLOAD_LEN + 2` bytes

---

## Система координат

### Система координат в метрах (реальный мир)

- **Начало координат:** Позиция камеры на автомобиле
- **X ось:** Вправо (положительные значения = вправо, отрицательные = влево)
- **Y ось:** Вперёд по направлению движения (всегда положительные)
- **Z ось:** Вверх (не используется для разметки на дороге)

**Единица измерения:** Метры (м)

### Преобразование координат

Координаты преобразуются из пикселей изображения в метры с использованием **матрицы перспективного преобразования** (`PerspectiveTransform.yaml`):

```python
# Преобразование пикселей в метры
[x', y', w'] = M * [x_px, y_px, 1]
x_meters = x' / w'
y_meters = y' / w'
```

где `M` - матрица перспективного преобразования 3x3.

---

## Типы сообщений (MSG_TYPE)

### 0x01 - LANE_LINES (Линии дорожной разметки)

**Размер payload:** `1 + N * 71` bytes (N линий)
**Общий размер кадра:** `12 + N * 71` bytes

Содержит информацию о линиях дорожной разметки с полиномиальными коэффициентами, тремя точками в метрах и их пиксельными координатами.

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| count | uint8 | 1 byte | Количество линий N |

**Для каждой линии (71 bytes):**

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| side | uint8 | 1 byte | Сторона линии: 0=unknown, 1=left, 2=right, 3=center |
| style | uint8 | 1 byte | Стиль: 0=unknown, 1=solid, 2=dashed, 3=double |
| color | uint8 | 1 byte | Цвет: 0=unknown, 1=white, 2=yellow, 3=red |
| poly_a | float32 | 4 bytes | Коэффициент 'a' полинома x = ay² + by + c |
| poly_b | float32 | 4 bytes | Коэффициент 'b' полинома |
| poly_c | float32 | 4 bytes | Коэффициент 'c' полинома |
| x_m | float32 | 4 bytes | Центр линии X в метрах |
| y_m | float32 | 4 bytes | Центр линии Y в метрах |
| point1_x_m | float32 | 4 bytes | Верхняя точка X в метрах |
| point1_y_m | float32 | 4 bytes | Верхняя точка Y в метрах |
| point2_x_m | float32 | 4 bytes | Средняя точка X в метрах |
| point2_y_m | float32 | 4 bytes | Средняя точка Y в метрах |
| point3_x_m | float32 | 4 bytes | Нижняя точка X в метрах |
| point3_y_m | float32 | 4 bytes | Нижняя точка Y в метрах |
| point1_x_px | float32 | 4 bytes | Верхняя точка X в пикселях (OpenCV: 0,0 = верхний левый угол) |
| point1_y_px | float32 | 4 bytes | Верхняя точка Y в пикселях (OpenCV: Y вниз) |
| point2_x_px | float32 | 4 bytes | Средняя точка X в пикселях |
| point2_y_px | float32 | 4 bytes | Средняя точка Y в пикселях |
| point3_x_px | float32 | 4 bytes | Нижняя точка X в пикселях |
| point3_y_px | float32 | 4 bytes | Нижняя точка Y в пикселях |

**Формат пакета struct:**
```python
# Header
struct.pack("<BBBIH", VERSION, 0x01, seq, timestamp, payload_len)

# Payload
struct.pack("<B", count)  # Count of lines
for each line:
    struct.pack("<BBBfffffffffffffffff",  # 3 bytes + 17 floats = 71 bytes
        side, style, color,
        poly_a, poly_b, poly_c,
        x_m, y_m,
        point1_x_m, point1_y_m,
        point2_x_m, point2_y_m,
        point3_x_m, point3_y_m,
        point1_x_px, point1_y_px,
        point2_x_px, point2_y_px,
        point3_x_px, point3_y_px)
```

**Пример использования:**

```python
# Получили линию:
line = {
    'side': 1,  # left
    'style': 1,  # solid
    'color': 1,  # white
    'poly_a': 0.0005,
    'poly_b': -0.3,
    'poly_c': 250.0,
    'x_m': 0.8,   # center X in meters
    'y_m': 15.0,  # center Y in meters
    'points_m': [
        (1.2, 5.0),   # top point (x, y) in meters
        (0.8, 15.0),  # middle point
        (0.5, 25.0)   # bottom point
    ],
    'points_px': [
        (225.0, 100.0),  # top point in pixels (OpenCV coordinates)
        (206.2, 250.0),  # middle point
        (210.0, 400.0)   # bottom point
    ]
}

# Точки в метрах можно использовать напрямую для отрисовки 3D сцены
# Точки в пикселях для отладки / overlay на исходное изображение
# Полином x = ay² + by + c позволяет генерировать дополнительные точки
```

---

### 0x02 - ROAD_OBJECTS (Объекты на дороге)

**Размер payload:** `1 + N * 25` bytes (N объектов)
**Общий размер кадра:** `12 + N * 25` bytes

Содержит информацию об объектах дорожной разметки (стрелки, пешеходные переходы, стоп-линии и т.д.)

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| count | uint8 | 1 byte | Количество объектов N |

**Для каждого объекта (25 bytes):**

| Поле | Тип | Размер | Описание |
|------|-----|--------|----------|
| class_id | uint8 | 1 byte | ID класса объекта (см. таблицу классов) |
| center_x | float32 | 4 bytes | Центр X в метрах |
| center_y | float32 | 4 bytes | Центр Y в метрах (вперёд от камеры) |
| length | float32 | 4 bytes | Длина объекта в метрах |
| width | float32 | 4 bytes | Ширина объекта в метрах |
| yaw | float32 | 4 bytes | Ориентация в радианах (0 = вперёд) |
| confidence | uint8 | 1 byte | Уверенность детектирования (0-255) |
| flags | uint8 | 1 byte | Флаги состояния (битовая маска) |
| reserved | uint16 | 2 bytes | Резерв для будущего использования |

**Формат пакета struct:**
```python
# Payload
struct.pack("<B", count)  # Count of objects
for each object:
    struct.pack("<BffffBBH",
        class_id,
        center_x, center_y,
        length, width,
        yaw,
        confidence, flags,
        reserved)
```

**Пример использования:**

```python
# Получили стрелку:
arrow = {
    'class_id': 12,  # arrow_straight
    'center_x': 0.2,  # 0.2 метра вправо от центра
    'center_y': 10.0,  # 10 метров вперёд
    'length': 3.5,  # 3.5 метра в длину
    'width': 1.2,  # 1.2 метра в ширину
    'yaw': 0.0,  # направлена вперёд (0 радиан)
    'confidence': 200,  # уверенность 200/255
    'flags': 0
}

# Отрисовка в 3D:
# - Позиция: (0.2, 10.0, 0)
# - Размер: 3.5 x 1.2 метров
# - Поворот: 0° (вперёд)
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

При каждом обновлении детектирования отправляется **2 сообщения** в следующем порядке:

1. **LANE_LINES** (0x01) - линии дорожной разметки с полиномами и точками
2. **ROAD_OBJECTS** (0x02) - объекты разметки (стрелки, переходы и т.д.)

**Частота:** ~10-30 Hz (зависит от скорости обработки)

---

## Пример парсинга на приборной панели

```python
import struct

def parse_frame(data: bytes):
    """Parse V2 protocol frame."""
    # Проверка синхробайта
    if data[0] != 0xAA:
        return None

    # Парсинг заголовка
    version, msg_type, seq, timestamp, payload_len = struct.unpack("<BBBIH", data[1:10])

    if version != 0x02:
        print(f"Warning: Expected version 2, got {version}")

    # Извлечение payload
    payload = data[10:10+payload_len]

    # Проверка CRC
    crc_received = struct.unpack("<H", data[10+payload_len:12+payload_len])[0]
    crc_calculated = crc16_ibm(data[1:10+payload_len])
    if crc_received != crc_calculated:
        return None

    # Парсинг в зависимости от типа
    if msg_type == 0x01:  # LANE_LINES
        return parse_lane_lines(payload)
    elif msg_type == 0x02:  # ROAD_OBJECTS
        return parse_road_objects(payload)
    else:
        print(f"Unknown message type: 0x{msg_type:02X}")
        return None

def parse_lane_lines(payload: bytes):
    """Parse LANE_LINES message."""
    count = struct.unpack("<B", payload[0:1])[0]
    lines = []

    offset = 1
    for i in range(count):
        line_data = struct.unpack("<BBBfffffffffff", payload[offset:offset+38])
        line = {
            'side': line_data[0],  # 0=unknown, 1=left, 2=right, 3=center
            'style': line_data[1],  # 0=unknown, 1=solid, 2=dashed, 3=double
            'color': line_data[2],  # 0=unknown, 1=white, 2=yellow, 3=red
            'poly_a': line_data[3],
            'poly_b': line_data[4],
            'poly_c': line_data[5],
            'points': [
                (line_data[6], line_data[7]),   # point1 (x, y) in meters
                (line_data[8], line_data[9]),   # point2
                (line_data[10], line_data[11])  # point3
            ]
        }
        lines.append(line)
        offset += 38

    return {'type': 'lane_lines', 'lines': lines}

def parse_road_objects(payload: bytes):
    """Parse ROAD_OBJECTS message."""
    count = struct.unpack("<B", payload[0:1])[0]
    objects = []

    offset = 1
    for i in range(count):
        obj_data = struct.unpack("<BffffBBH", payload[offset:offset+25])
        obj = {
            'class_id': obj_data[0],
            'center_x': obj_data[1],  # meters
            'center_y': obj_data[2],  # meters
            'length': obj_data[3],    # meters
            'width': obj_data[4],     # meters
            'yaw': obj_data[5],       # radians
            'confidence': obj_data[6],
            'flags': obj_data[7],
            'reserved': obj_data[8]
        }
        objects.append(obj)
        offset += 25

    return {'type': 'road_objects', 'objects': objects}

def crc16_ibm(data: bytes) -> int:
    """CRC16-IBM calculation."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
            crc &= 0xFFFF
    return crc
```

---

## Отрисовка 3D сцены на приборной панели

### Шаг 1: Получение данных

```python
# Получаем 2 типа сообщений
lane_lines_data = parse_frame(lane_lines_frame)
road_objects_data = parse_frame(road_objects_frame)
```

### Шаг 2: Отрисовка линий разметки

```python
for line in lane_lines_data['lines']:
    # Координаты уже в метрах!
    points_3d = [
        (p[0], p[1], 0.0) for p in line['points']
    ]

    # Отрисовка в зависимости от стиля
    if line['style'] == 1:  # SOLID
        draw_solid_line_3d(points_3d, color=get_color(line['color']), width=0.1)
    elif line['style'] == 2:  # DASHED
        draw_dashed_line_3d(points_3d, color=get_color(line['color']), width=0.1)
    elif line['style'] == 3:  # DOUBLE
        draw_double_line_3d(points_3d, color=get_color(line['color']), offset=0.15)
```

### Шаг 3: Отрисовка объектов

```python
for obj in road_objects_data['objects']:
    position_3d = (obj['center_x'], obj['center_y'], 0.0)

    if obj['class_id'] == 12:  # arrow_straight
        draw_arrow_3d(
            position=position_3d,
            yaw=obj['yaw'],
            length=obj['length'],
            width=obj['width']
        )
    elif obj['class_id'] == 2:  # crosswalk
        draw_crosswalk_3d(
            position=position_3d,
            size=(obj['length'], obj['width']),
            yaw=obj['yaw']
        )
```

---

## Преимущества V2 протокола

✅ **Упрощение:** Только 2 типа сообщений вместо 5
✅ **Точность:** Все координаты в метрах с использованием калибровки камеры
✅ **Согласованность:** Отправляются только данные, которые реально рисуются на экране
✅ **Система координат:** Понятная система (Y вперёд, X вправо)
✅ **Меньше трафика:** Убраны избыточные сообщения
✅ **Готовность к 3D:** Координаты готовы для прямой отрисовки в 3D пространстве

---

## Частые вопросы (FAQ)

**Q: Почему координаты в метрах, а не в пикселях?**
A: Метры - это реальные расстояния на дороге. Пиксели зависят от разрешения камеры и перспективы. Для приборной панели важны реальные расстояния.

**Q: Как часто приходят сообщения?**
A: ~10-30 Hz, зависит от производительности. Каждый кадр генерирует 2 сообщения подряд.

**Q: Нужно ли буферизовать сообщения?**
A: Рекомендуется хранить только последнее значение каждого типа для минимальной задержки.

**Q: Что делать если координаты выходят за границы?**
A: Отбрасывать или обрезать объекты, выходящие за видимую область на приборной панели.

**Q: Какая точность координат?**
A: Зависит от качества калибровки камеры. При хорошей калибровке точность ~5-10 см.

---

## История изменений

- **v1.0** - Базовые сообщения (0x01-0x04)
- **v1.1** - Добавлено сообщение FITTED_LINES (0x05)
- **v2.0** - Упрощение до 2 типов сообщений, координаты в метрах
