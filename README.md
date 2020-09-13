# DeepMorphy #
[![Nuget](https://img.shields.io/nuget/v/DeepMorphy.svg?style=popout)](https://www.nuget.org/packages/DeepMorphy/)

DeepMorphy is a neural network based morphological analyzer for Russian language.
___
DeepMorphy - морфологический анализатор для русского языка. Доступен как .Net Standart 2.0 библиотека.
Умеет:
 1. проводить морфологический разбор слова (определяет часть речи, род, число, падеж, время, лицо, наклонение, залог);
 2. приводить слова к нормальной форме;
 3. менять форму слова в рамках лексемы. 

* [Терминология](#терминология)
* [Принцип работы](#принцип-работы)
    * [Препроцессоры](#препроцессоры)
    * [Нейронная сеть](#нейронная-сеть)
* [Руководство пользователя](#руководство-пользователя)
    * [Установка](#установка)
    * [Начало использования](#начало-использования)
    * [Морфологический разбор](#морфологический-разбор)
    * [Лемматизация](#лемматизация)
    * [Изменение формы слова](#изменение-формы-слова)
* [Структура репозитория](#cтруктура-репозитория)
* [Планы по доработкам](#планы-по-доработкам)




## Терминология
Терминология в DeepMorphy частично заимствована из морфологического анализатора [pymorphy2](https://pymorphy2.readthedocs.io/en/latest/).

**Граммема** (англ. grammeme) - значение одной из грамматических категорий слова (например прошедшее время, 
единственное число, мужской род).

**Грамматическая категория** (англ. grammatical category) - множество из взаимоисключающих друг друга граммем, 
характеризующих какой-то общий признак (например род, время, падеж и тп). Список всех поддерживаемых в DeepMorphy 
категорий и граммем [тут](gram.md).

**Тег** (англ. tag) - набор граммем, характеризующих данное слово
 (например, тег для слова еж - существительное, единственное число, именительный падеж, мужской род). 

**Лемма** (англ. lemma) - нормальная форма слова. 

**Лемматизация** (англ. lemmatization) - приведение слова к нормальной форме.

**Лексема** - набор всех форм одного слова.


## Принцип работы

Основным элементом DeepMorphy является нейронная сеть. Для большинства слов морфологический анализ и лемматизация
выполняется сетью. Некоторые виды слов обрабатываются препроцессорами.

### Препроцессоры

Имеется 3 препроцессора:
* Словарь. Часть токенов просто смотрится в словаре. 
Используется для местоимений, предикативов, предлогов, союзов, частиц, междометий и числительных.  Так же в словарь добавляются слова из датасета, в которых сеть после обучения все еще делает ошибки.
* Препроцессор для наращенных числительных (например 1-й, 1917-й).
* Препроцессор на регулярных выражениях для пунктуации, целых цифр, римских цифр и неизвестных токенов 
(если токен в основном состоит не из кириллицы).

### Нейронная сеть

Сеть построена и обучена на фреймворке [tensorflow](https://www.tensorflow.org/). 
В качестве датасета выступает словарь [Opencorpora](http://opencorpora.org/). В .Net интегрирована через 
[TensorFlowSharp](https://github.com/migueldeicaza/TensorFlowSharp).


Граф вычислений для разбора слов в DeepMorphy состоит из 10 "подсетей":
* 8 двунаправленных рекурентных сетей, по одной для каждой поддерживаемой грамматической категории 
(определяет граммему в категории);
* 1 двунаправленная рекурентная сеть для определения самых вероятных тегов. Для каждой комбинации граммем из датасета заведен
1 класс (всего 232 класса), сеть обучается на определение к каким классам может принадлежать данное слово. На этапе
работы берется 4 самых вероятных класса;
* 1 seq2seq модель для лемматизации.

Задача изменения формы слов решается 1 seq2seq сетью.
 
![Примерная схема сети для разбора слов](network.png)


Обучение сетей производится последовательно, сначала обучаются сети по категориям 
(порядок не имеет значения). Далее обучается главная классификация по тегам, лемматизация и сеть для изменения формы слов.
Обучение проводилось на 3-ех GPU Titan X. Метрики работы сети на тестовой датасете для последнего релиза можно посмотреть 
[тут](https://github.com/lepeap/DeepMorphy/blob/master/src/py/model/latest_release/test_info.txt).




## Руководство пользователя

DeepMorphy для .NET представляет собой библиотеку .Net Standart 2.0. В зависимостях только библиотека [TensorflowSharp](https://github.com/migueldeicaza/TensorFlowSharp) (через нее запускается нейронная сеть).

### Установка

Библиотека опубликована в [Nuget](https://www.nuget.org/packages/DeepMorphy/), поэтому проще всего устанавливать через него.

Если есть менеджер пакетов:
```
 Install-Package DeepMorphy
```
Если проект поддерживает PackageReference:
```
 <PackageReference Include="DeepMorphy"/> 
```
Если кто-то хочет собрать из исходников, то C# исходники лежат [тут](https://github.com/lepeap/DeepMorphy/tree/master/src/cs). 
Для разработки используется Rider (без проблем все должно собраться и в студии).

### Начало использования
Все действия осуществляются через объект класса MorphAnalyzer:
```csharp
var morph = new MorphAnalyzer();
```
В идеале, лучше использовать его как синглтон, при создании объекта какое-то время уходит на загрузку словарей и сети. Потокобезопасен. При создании в конструктор можно передать следующие параметры:
* **withLemmatization** - возвращать ли леммы при разборе слов (по умолчанию - false). Если нужна лемматизация при разборе, то необходимо выставить в true, иначе лучше не включать (без флага работает быстрее).
* **useEnGrams** - использовать английские названия граммем и грамматических категорий (по умолчанию - false).
* **withTrimAndLower** - производить ли обрезку пробелов и приведение слов к нижнему регистру (по умолчанию - true).
* **maxBatchSize** - максимальный батч, который скармливается нейронной сети (по умолчанию - 4096). 
Если есть уверенность, что оперативной памяти хватит, то можно увеличивать (увеличит скорость обработки для большого количества слов).
Примеры использования [тут](https://github.com/lepeap/DeepMorphy/blob/master/src/cs/ExampleConsole/Program.cs).

### Морфологический разбор
Для разбора используется метод Parse (на вход принимает IEnumerable<string> со словами для анализа, возвращает IEnumerable<MorphInfo> с результатом анализа).
```csharp
var results = morph.Parse(new string[]
{
    "королёвские",
    "тысячу",
    "миллионных",
    "красотка",
    "1-ый"
}).ToArray();
var morphInfo = results[0];
```
Список поддерживаемых грамматических категорий, граммем и их ключей [тут](gram.md).
Если необходимо узнать самую вероятную комбинацию граммем (тег), то нужно использовать свойство BestTag объекта MorphInfo.
```csharp
// выводим лучшую комбинацию граммем для слова
Console.WriteLine(morphInfo.BestTag);
```
По самому слову не всегда возможно однозначно установить значения его грамматических категорий 
(см. [омонимы](https://ru.wikipedia.org/wiki/%D0%9E%D0%BC%D0%BE%D0%BD%D0%B8%D0%BC%D1%8B)),
 поэтому DeepMorphy позволяет посмотреть топ тегов для данного слова (свойство Tags).
```csharp
// выводим все теги для слова + их вероятность
foreach (var tag in morphInfo.Tags)
    Console.WriteLine($"{tag} : {tag.Power}");
```
Есть ли комбинация граммем в каком-нибудь из тегов:
```csharp
// есть ли в каком-нибудь из тегов прилагательные единственного числа
morphInfo.HasCombination("прил", "ед");
```
Есть ли комбинация граммем в самом вероятном теге:
```csharp
// ясляется ли лучший тег прилагательным единственного числа
morphInfo.BestTag.Has("прил", "ед");
```
Получение определенных из лучшего тега грамматических категорий:
```csharp
// выводит часть речи лучшего тега и число
Console.WriteLine(morphInfo.BestTag["чр"]);
Console.WriteLine(morphInfo.BestTag["число"]);
```

Теги применяются для случаев, если нужна информация сразу о нескольких грамматических категориях (например часть речи и число).
Если вас интересует только одна категория, то можно использовать интерфейс к 
вероятностям значений грамматических категорий объектов MorphInfo.

```csharp
// выводит самую вероятную часть речи
Console.WriteLine(morphInfo["чр"].BestGramKey);
```
Так же можно получить распределение вероятностей по грамматической категории:
```csharp
// выводит распределение вероятностей для падежа
foreach (var gram in morphInfo["падеж"].Grams)
{
    Console.WriteLine($"{gram.Key}:{gram.Power}");
}
```

### Лемматизация

Если вместе с морфологическим анализом нужно получать леммы слов, то анализатор надо создавать следующим образом:
```csharp
var morph = new MorphAnalyzer(withLemmatization: true);
```
Леммы можно получить из тегов слова:
```csharp
Console.WriteLine(morphInfo.BestTag.Lemma);
```
Проверка, есть ли у данного слова лемма:
```csharp
morphInfo.HasLemma("королевский");
```
Метод CanBeSameLexeme может быть использован для нахождения слов одной лексемы:
```csharp
// выводим все слова, которые могут быть формой слова королевский
var words = new string[]
{
    "королевский",
    "королевские",
    "корабли",
    "пересказывают",
    "королевского"
};

var results = morph.Parse(words).ToArray();
var mainWord = results[0];
foreach (var morphInfo in results)
{
    if (mainWord.CanBeSameLexeme(morphInfo))    
        Console.WriteLine(morphInfo.Text);
}
```
Если необходима только лемматизация без морфологического разбора, то нужно использовать метод Lemmatize:
```csharp
var tasks = new []
{
    new LemTask("синяя", morph.TagHelper.CreateTag("прил", gndr: "жен", nmbr: "ед", @case: "им")),
    new LemTask("гуляя", morph.TagHelper.CreateTag("деепр", tens: "наст"))
};

var lemmas = morph.Lemmatize(tasks).ToArray();
foreach (var lemma in lemmas)
{
    Console.WriteLine(lemma);
}
```
### Изменение формы слова
DeepMorphy умеет изменять форму слова в рамках лексемы, список возможных поддерживаемых словоизменений [тут](inflect.md). 
Словарные слова возможно изменять только в рамках тех форм, которые доступны в словаре.
Для изменения формы слов используется метод Inflect, 
на вход принимает перечисление объектов InflectTask (содержит исходное слово, тег исходного слова и тег, в который слово нужно поставить).
На выходе перечисление с требуемыми формами (если форму не удалось обработать, то null).
```csharp
var tasks = new[]
{
    new InflectTask("синяя", 
        morph.TagHelper.CreateTag("прил", gndr: "жен", nmbr: "ед", @case: "им"),
        morph.TagHelper.CreateTag("прил", gndr: "муж", nmbr: "ед", @case: "им")),
    new InflectTask("гулять", 
        morph.TagHelper.CreateTag("инф_гл"),  
        morph.TagHelper.CreateTag("деепр", tens: "наст"))
};

var results = morph.Inflect(tasks);
foreach (var result in results)
{
    Console.WriteLine(result);
}
```
Так же для слова имеется возможность получить все его формы с помощью метода Lexeme 
(для словарных слов возвращает все из словаря, для остальных все формы из [поддерживаемых словоизменений](inflect.md)).
```csharp
var word = "лемматизировать";
var tag = m.TagHelper.CreateTag("инф_гл");
var results = m.Lexeme(word, tag).ToArray();
```
Одной из особенностей алгоритма является то, что при изменении формы или генерации лексемы, 
сеть может "выдумать" несуществующую (гипотетическую) форму слова, форму которая не употребляется в языке. Например, 
ниже получится слово "побежу", хотя в данный момент в языке оно не особо используется.
```csharp
var tasks = new[]
{
    new InflectTask("победить", 
        m.TagHelper.CreateTag("инф_гл"),  
        m.TagHelper.CreateTag("гл", nmbr: "ед", tens: "буд", pers: "1л", mood: "изъяв"))
};
Console.WriteLine(m.Inflect(tasks).First());
```
## Cтруктура репозитория
* [Python код модели, обучения и разные утилиты](https://github.com/lepeap/DeepMorphy/tree/master/src/py/model).
* [C# код DeepMorphy](https://github.com/lepeap/DeepMorphy/tree/master/src/cs).
* [Проект с C# примерами](https://github.com/lepeap/DeepMorphy/tree/master/src/cs/ExampleConsole).
* [Файлы последнего релиза](https://github.com/lepeap/DeepMorphy/tree/master/src/py/model/latest_release).

## Планы по доработкам
* Подумать над оптимизацией модели на этапе применения (подумать над квантованием или обрезкой графа вычислений).

