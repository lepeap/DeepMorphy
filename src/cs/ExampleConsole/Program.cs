using System;
using System.Linq;
using DeepMorphy;
using DeepMorphy.Model;

namespace ExampleConsole
{
    class Program
    {
        private static string[] Words = new string[]
        {
            "tafsdfdfasd",
            "xii",
            "123",
            ".345",
            "43,34",
            "..!",
            "1-ый",
            "бутявка",
            "пальто",
            "плакса",
            "в",
            "действуя",
            "королёвские",
            "большая",
            "двадцать",
            "тысячу",
            "миллионных",
            "222-ого",
            "дотошный",
            "красотка",
            "центральные",
            "укрывал",
            "шоссе"
        };

        static void Main(string[] args)
        {
            Simple();

            ParseFull1();
            ParseFull2();
            ParseFull3();
            ParseFull4();
            ParseFull5();

            ParsePart1();
            ParsePart2();
            ParsePart3();

            Lemmatization1();
            Lemmatization2();
            Lemmatization3();

            Inflect1();
            Inflect2();
            Inflect3();
            
            Lexeme1();
            Lexeme2();
        }

        static void Simple()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();

            foreach (var morphInfo in results)
            {
                Console.WriteLine(morphInfo.ToString());
            }
        }

        static void ParseFull1()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            WriteHeader("Лучший тег");
            foreach (var morphInfo in results)
            {
                Console.WriteLine($"{morphInfo.Text} - {morphInfo.BestTag}");
            }
        }

        static void ParseFull2()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            WriteHeader("Все топ теги");
            foreach (var morphInfo in results)
            {
                Console.WriteLine($"{morphInfo.Text}:");
                foreach (var tag in morphInfo.Tags)
                {
                    Console.WriteLine($"    {tag} : {tag.Power}");
                }
            }
        }

        static void ParseFull3()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            WriteHeader("Теги с прилагательным и единственным числом");
            foreach (var morphInfo in results)
            {
                foreach (var tag in morphInfo.Tags)
                {
                    if (tag.Has("прил", "ед"))
                    {
                        Console.WriteLine($"{morphInfo.Text} {tag} : {tag.Power}");
                    }
                }
            }
        }

        static void ParseFull4()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            WriteHeader("Вывод только части речи и числа");
            foreach (var morphInfo in results)
            {
                Console.WriteLine($"{morphInfo.Text}:");
                foreach (var tag in morphInfo.Tags)
                {
                    Console.WriteLine($"    {tag["чр"]} {tag["число"]}");
                }
            }
        }

        static void ParseFull5()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            WriteHeader("Слова, которые вероятно являются глаголами прошедшего времени");
            foreach (var morphInfo in results)
            {
                if (morphInfo.HasCombination("гл", "прош"))
                {
                    Console.WriteLine($"{morphInfo.Text}");
                }
            }
        }

        static void ParsePart1()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            WriteHeader("Только прилагательные");
            foreach (var morphInfo in results)
            {
                if (morphInfo["чр"].BestGramKey == "прил")
                {
                    Console.WriteLine(morphInfo.ToString());
                }
            }
        }

        static void ParsePart2()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            WriteHeader("Только лучшая часть речи с ее вероятностью");
            foreach (var morphInfo in results)
            {
                var bestGram = morphInfo["чр"].BestGram;
                Console.WriteLine($"{morphInfo.Text} - {bestGram.Key}:{bestGram.Power} ");
            }
        }

        static void ParsePart3()
        {
            var m = new MorphAnalyzer();
            WriteHeader("Полная информация по падежу");
            var results = m.Parse("речка").ToArray();

            foreach (var morphInfo in results)
            {
                Console.WriteLine(morphInfo.Text);
                foreach (var gram in morphInfo["падеж"].Grams)
                {
                    Console.WriteLine($"{gram.Key}:{gram.Power}");
                }
            }
        }

        static void Lemmatization1()
        {
            var m = new MorphAnalyzer(withLemmatization: true);
            WriteHeader("Выводим формы слова 'королевский'");

            var words = new string[]
            {
                "королевский",
                "королевские",
                "корабли",
                "укрывал",
                "обновляя",
                "выходящие",
                "собаковод",
                "раскладывала",
                "обучает",
                "юбка",
                "пересказывают",
                "королевского"
            };

            var results = m.Parse(words).ToArray();
            var mainWord = results[0];
            foreach (var morphInfo in results)
            {
                if (mainWord.CanBeSameLexeme(morphInfo))
                {
                    Console.WriteLine(morphInfo.Text);
                }
            }
        }

        static void Lemmatization2()
        {
            var m = new MorphAnalyzer(withLemmatization: true);
            WriteHeader("Выводим все леммы из главных тэгов");

            var words = new string[]
            {
                "королевские",
                "корабли",
                "укрывал",
                "обновляя",
                "выходящие",
                "собаковод",
                "раскладывала",
                "обучает",
                "юбка",
                "пересказывают",
                "шоссе"
            };

            var results = m.Parse(words).ToArray();

            foreach (var morphInfo in results)
            {
                Console.WriteLine(morphInfo.BestTag.Lemma);
            }
        }

        static void Lemmatization3()
        {
            var m = new MorphAnalyzer(withLemmatization: true);
            WriteHeader("Лемматизация без классификации");
            var tasks = new[]
            {
                new LemTask("синяя", m.TagHelper.CreateTag("прил", gndr: "жен", nmbr: "ед", @case: "им")),
                new LemTask("гуляя", m.TagHelper.CreateTag("деепр", tens: "наст"))
            };

            var lemmas = m.Lemmatize(tasks).ToArray();
            for (int i = 0; i < tasks.Length; i++)
            {
                Console.WriteLine($"{tasks[i].word} - {lemmas[i]}");
            }
        }

        static void Inflect1()
        {
            var m = new MorphAnalyzer(withLemmatization: true);
            WriteHeader("Изменение формы слов");
            var tasks = new[]
            {
                new InflectTask("синяя", 
                    m.TagHelper.CreateTag("прил", gndr: "жен", nmbr: "ед", @case: "им"),
                    m.TagHelper.CreateTag("прил", gndr: "муж", nmbr: "ед", @case: "им")),
                new InflectTask("гулять", 
                    m.TagHelper.CreateTag("инф_гл"),  
                    m.TagHelper.CreateTag("деепр", tens: "наст"))
            };

            var results = m.Inflect(tasks).ToArray();
            for (int i = 0; i < tasks.Length; i++)
            {
                Console.WriteLine($"{tasks[i].word} -> {results[i]} {tasks[i].resultTag}");
            }
        }
        
        static void Inflect2()
        {
            var m = new MorphAnalyzer(withLemmatization: true);
            WriteHeader("Переводим слова во множественное число");

            var morphRes = m.Parse("стула", "стола", "горшка").ToArray();

            var tasks = morphRes
                .Select(mi => new InflectTask(mi.Text, 
                                              mi.BestTag,
                                              m.TagHelper.CreateTag("сущ", gndr: mi.BestTag["род"], @case: mi.BestTag["падеж"], nmbr: "мн")))
                .ToArray();

            var results = m.Inflect(tasks).ToArray();
            for (int i = 0; i < tasks.Length; i++)
            {
                Console.WriteLine($"{tasks[i].word} {tasks[i].wordTag} -> {results[i]} {tasks[i].resultTag}");
            }
        }
        
        static void Inflect3()
        {
            var m = new MorphAnalyzer(withLemmatization: true);
            WriteHeader("Гипотетическая форма слова");
            var tasks = new[]
            {
                new InflectTask("победить", 
                    m.TagHelper.CreateTag("инф_гл"),  
                    m.TagHelper.CreateTag("гл", nmbr: "ед", tens: "буд", pers: "1л", mood: "изъяв"))
            };
            Console.WriteLine($"{tasks[0].word} {tasks[0].wordTag} -> {m.Inflect(tasks).First()} {tasks[0].resultTag}");
        }
        
        static void Lexeme1()
        {
            var m = new MorphAnalyzer(withLemmatization: true);
            var word = "лемматизировать";
            var tag = m.TagHelper.CreateTag("инф_гл");
            var results = m.Lexeme(word, tag).ToArray();

            WriteHeader($"Лексема для слова {word}[{tag}]");
            foreach (var tpl in results)
            {
                Console.WriteLine($"{tpl.tag} - {tpl.text}");
            }

            Console.WriteLine();

            WriteHeader($"Только деепричастия из лексемы {word}[{tag}]");
            foreach (var tpl in results.Where(x => x.tag.Has("деепр")))
            {
                Console.WriteLine($"{tpl.tag} - {tpl.text}");
            }
        }

        static void Lexeme2()
        {
            var m = new MorphAnalyzer(withLemmatization: true);
            var word = "я";
            var res = m.Parse("я").ToArray();
            var tag = res[0].BestTag;
            var results = m.Lexeme(word, tag).ToArray();
            WriteHeader($"Лексема для слова {word}[{tag}]");
            foreach (var tpl in results)
            {
                Console.WriteLine($"{tpl.tag} - {tpl.text}");
            }

            Console.WriteLine();
        }

        static void WriteHeader(string message)
        {
            Console.WriteLine();
            Console.WriteLine("####################################################");
            Console.WriteLine("####################################################");
            Console.WriteLine($"{message}:");
        }
    }
}