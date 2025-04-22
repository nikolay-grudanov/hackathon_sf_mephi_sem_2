hackathon_sf_mephi_sem_2
==============================

Хакатон за второй семестр

## Задача

**Задача 2: Семантический поиск фраз в текстах**

Создать систему семантического текстового поиска слов (словосочетаний) в документах, учитывающую не только точное написание, но и смысловое значение.

Результатом должно быть определение позиции найденного слова/словосочетания в тексте и оценка вероятности совпадения.

## Команда проекта

1. Груданов Николай Алексеевич
2. Душкин Василий Алексеевич
3. Мусатов Матвей Геннадьевич
4. Комаров Василий Владимирович
5. Черников Дмитрий Сергеевич

Структура проекта
------------

Был использован шаблон cookiecutter


```
hackathon_sf_mephi_sem_2/  
├── LICENSE     
├── README.md                  
├── Makefile                     # Makefile с командами вроде `make data` или `make train`                   
├── configs                      # Файлы конфигурации (модели и гиперпараметры обучения)
│   └── model1.yaml              
│
├── data                         
│   ├── external                 # Данные из сторонних источников.
│   ├── interim                  # Промежуточные данные, которые были преобразованы.
│   ├── processed                # Финальные канонические наборы данных для моделирования.
│   └── raw                      # Исходные неизменяемые данные.
│
├── docs                         # Документация проекта.
│
├── models                       # Обученные сериализованные модели.
│
├── notebooks                    # Jupyter-ноутбуки.
│
├── references                   # Словари данных, руководства и другие пояснительные материалы.
│
├── reports                      # Сгенерированные отчеты в форматах HTML, PDF, LaTeX и др.
│   └── figures                  # Графики и рисунки для отчетов.
│
├── requirements.txt             # Файл требований для воспроизведения среды анализа.
└── src                          # Исходный код проекта.
    ├── __init__.py              # Делает src Python-модулем.
    │
    ├── data                     # Скрипты для обработки данных.
    │   ├── build_features.py    # Построение признаков
    │   ├── cleaning.py          # Очистка данных
    │   ├── ingestion.py         # Загрузка данных
    │   ├── labeling.py          # Маркировка данных
    │   ├── splitting.py         # Разделение данных
    │   └── validation.py        # Валидация данных
    │
    ├── models                   # Инженерия ML-моделей (отдельная папка для каждой модели).
    │   └── model1      
    │       ├── dataloader.py    # Загрузчик данных
    │       ├── hyperparameters_tuning.py # Настройка гиперпараметров
    │       ├── model.py         # Определение модели
    │       ├── predict.py       # Скрипт для предсказаний
    │       ├── preprocessing.py # Предобработка данных
    │       └── train.py         # Обучение модели
    │
    └── visualization            # Скрипты для визуализации данных.
        ├── evaluation.py        # Визуализация результатов
        └── exploration.py       # Исследовательский анализ
```

--------
<p><small>Project based on the <a target="_blank" href="https://github.com/Chim-SO/cookiecutter-mlops/">cookiecutter MLOps project template</a>
that is originally based on <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 
#cookiecuttermlops #cookiecutterdatascience</small></p>
