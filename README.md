# Проект Машинного Зрения

## Описание

Этот проект представляет собой систему машинного зрения, основанную на нейронных сетях и оптимизированную для использования на платформе NanoPi Neo Air. Проект включает в себя обучение нейросети с использованием модели MobileSSD для распознавания объектов в режиме реального времени.

## Структура проекта

- **logs/**: Директория для публикации логов.
- **parts-of-app/**: Исходный код проекта по функциям.
- **pictures/**: картинки для отладки кода.
- **python/**: Код для инференса на платформе NanoPi Neo Air.
- **camWeb/**: Главная папка с кодом программы на Flask. 
- **scripts/**: Код для автозапуска.
- **trash/**: Старые шаблоны на html.

## Инструкции по установке

1. Клонируйте репозиторий:

    ```bash
    git clone https://github.com/ваш-локальный-путь/flask-приложение.git
    cd flask-приложение
    ```

2. Установите зависимости:

    ```bash
    pip3 install -r requirements.txt
    ```
    Для того, чтобы в интерфейсе работала командная строка из браузера, то установите пакет по инструкции https://www.tecmint.com/shellinabox-web-based-ssh-linux-terminal/

## Запуск приложения

1. Перейдите в директорию с приложением:

    ```bash
    cd python/camWeb
    ```

2. Запустите Flask приложение:

    ```bash
    python3 app.py
    ```

3. Откройте веб-браузер и перейдите по адресу [http://localhost:5050](http://localhost:5050) для доступа к приложению.

## Использование

- **Главная страница:**
  - По адресу [http://localhost:5050](http://localhost:5050) вы увидите главную страницу.




