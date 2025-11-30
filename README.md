# NN Line Detector

Qt-приложение для детекции дорожной разметки: читает RTSP/UDP видеопоток, обрабатывет кадры через сегментационную нейросеть, постобрабатывает и отправляет данные по TCP в виде протокольных кадров LaneSummary и MarkingObjects.

## Быстрый старт
1. Установить зависимости (рекомендуется venv):
   ```bash
   pip install -r requirements.txt
   ```
2. Убедиться, что в системе установлены GStreamer + PyGObject и Qt/X11 зависимости (libxcb и т.п.).
3. Настроить `config/config.yaml`:
   - `rtsp.uri` — источник потока (например, `udp://239.0.0.1:5000`).
   - `nn.model_path` — путь к модели (`NN/model_traced.pt`).
   - `tcp.host`/`tcp.port` — получатель протокольных сообщений.
   - `render.enabled` — включать/выключать отрисовку видео в UI.
4. Запуск:
   ```bash
   python app.py
   ```

## Кратко об архитектуре
- `core/video_capture.py` — RTSP/UDP вход на GStreamer.
- `core/nn_engine.py` — адаптер модели (TorchScript/PyTorch), preprocessing -> logits -> маски/bbox.
- `core/postprocess.py` — фильтрация/мердж, построение LaneSummary и MarkingObject.
- `core/renderer.py` — рисует маски/боксы (можно отключить).
- `core/inference_worker.py` — поток инференса, отдаёт кадры/данные в UI.
- `network/protocol.py` — сборка кадров LaneSummary/MarkingObjects (CRC16 IBM, seq, timestamp).
- `network/tcp_client.py` — TCP-клиент с очередью и переподключением.
- `ui/main_window.py` — окно Qt с контролами, статусами и видео.

## Примечания
- Чтобы не держать большие веса в репозитории, файлы `NN/*.pth|*.pt` игнорируются. Храните модели локально или через Git LFS/артефакты.
- Для отладки TCP можно поднять простой сервер: `nc -l 9000` и смотреть сырые байты.
