# citizen-requests



## Код обучения и инференса команды Prime99

Модели по данной ссылке на Яндекс.Диск нужно распаковать в папку `ml_fastapi`
https://disk.yandex.ru/d/BpqU6XN6NKwfWA

Код моделей лежит в папке notebooks


## Особенности подхода

   разделили тренировочный датасет на тренировочную, валидационную и тестовую выборки
   пробовали разные подходы:
        сопоставление с метками классов на основе текстовых и векторных признаков
        обучение CatBoost моделей
        обучение классификаторов на основе модели xlm-roberta-base
        обучение на основе подхода metric learning с использованием алгоритма SetFIt      (пробовали rubert2-tiny и labse-en-ru модели)

    для полей "Исполнитель" и "Группа тем" наибольший результат на нашей тестовой выборке дал ансамбль Catboost + Setfit, для "Тема" - SetFit.

Использовали свободное ПО transformers и catboost.

Бэкенд сервер работает на основе открытой библиотеки FastAPI.

Также мы экспериментировали с иерархической классификацией, но в конечной версии решили остановитьсяна единократном прогоне модели.


Из плюсов нашего подхода:
    так как мы использовали подход metric learning, то добавление новых классов возможно без дообучения самой модели
    модели являются легковесными и легко применимы как на CPU, так и на GPU
    для новых классов достаточно всего 10 текстов
