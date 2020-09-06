using System;
using System.Linq;
using DeepMorphy;

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
            SimpleExample();
            
            AnalisysFullExample1();
            AnalisysFullExample2();
            AnalisysFullExample3();
            AnalisysFullExample4();
            AnalisysFullExample5();
            
            AnalisysPartExample1();
            AnalisysPartExample2();
            AnalisysPartExample3();
            
            LemmatizationExample1();
            LemmatizationExample2();
        
            Lexeme();
            Lexeme1();
        }

        static void SimpleExample()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();

            foreach (var morphInfo in results)
            {
                Console.WriteLine(morphInfo.ToString());
            }
        }
        
        static void AnalisysFullExample1()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            WriteHeader("Лучший тег");
            foreach (var morphInfo in results)
            {
                Console.WriteLine($"{morphInfo.Text} - {morphInfo.BestTag}");
            }
        }
        
        static void AnalisysFullExample2()
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
        
        static void AnalisysFullExample3()
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
        
        static void AnalisysFullExample4()
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
        
        static void AnalisysFullExample5()
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
                
        static void AnalisysPartExample1()
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

        static void AnalisysPartExample2()
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
        
        static void AnalisysPartExample3()
        {
            var m = new MorphAnalyzer();
            WriteHeader("Полная информация по падежу");
            var results = m.Parse(new string[]{"речка"}).ToArray();

            foreach (var morphInfo in results)
            {
                Console.WriteLine(morphInfo.Text);
                foreach (var gram in morphInfo["падеж"].Grams)
                {
                    Console.WriteLine($"{gram.Key}:{gram.Power}");
                }
            }
        }
        
        static void LemmatizationExample1()
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

                
        static void LemmatizationExample2()
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

        static void Lexeme()
        {
            var m = new MorphAnalyzer(withLemmatization: true);
            var word = "дебажить";
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
        
        
        static void Lexeme1()
        {
            var m = new MorphAnalyzer(withLemmatization: true);
            var word = "я";
            var res = m.Parse("я").ToArray();
            var tag = res[0].BestTag;
            var results = m.Lexeme(word,tag ).ToArray();
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