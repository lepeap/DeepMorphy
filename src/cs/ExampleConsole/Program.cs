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
            "укрывал"
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
        }

        static void SimpleExample()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();

            foreach (var morphInfo in results)
                Console.WriteLine(morphInfo.ToString());
        }
        
        static void AnalisysFullExample1()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            Console.WriteLine("Лучший тег");
            foreach (var morphInfo in results)
                Console.WriteLine($"{morphInfo.Text} - {morphInfo.BestTag}");
        }
        
        static void AnalisysFullExample2()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            Console.WriteLine("Все топ теги");
            foreach (var morphInfo in results)
            {
                Console.WriteLine($"{morphInfo.Text}:");
                foreach (var tag in morphInfo.Tags)
                    Console.WriteLine($"    {tag} : {tag.Power}");
            }
        }
        
        static void AnalisysFullExample3()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            Console.WriteLine("Теги с прилагательным и единственным числом");
            foreach (var morphInfo in results)
            {
                foreach (var tag in morphInfo.Tags)
                    if (tag.Has("прил", "ед"))
                        Console.WriteLine($"{morphInfo.Text} {tag} : {tag.Power}");
            }
        }
        
        static void AnalisysFullExample4()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            Console.WriteLine("Вывод только части речи и числа");
            foreach (var morphInfo in results)
            {
                Console.WriteLine($"{morphInfo.Text}:");
                foreach (var tag in morphInfo.Tags)
                    Console.WriteLine($"    {tag["чр"]} {tag["число"]}");
            }
        }
        
        static void AnalisysFullExample5()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            Console.WriteLine("Слова, которые вероятно являются глаголами прошедшего времени");
            foreach (var morphInfo in results)
            {
                if (morphInfo.HasCombination("гл", "прош"))
                    Console.WriteLine($"{morphInfo.Text}");
            }
        }
                
        static void AnalisysPartExample1()
        {
            var m = new MorphAnalyzer();
            var results = m.Parse(Words).ToArray();
            WriteHeader("Только прилагательные");
            foreach (var morphInfo in results)
                if (morphInfo["чр"].BestGramKey=="прил")
                    Console.WriteLine(morphInfo.ToString());
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
                "королевские",
                "корабли",
                "укрывал",
                "обновляя",
                "выходящие",
                "собаковод",
                "раскладывала", 
                "обучает",
                "юбка",
                "шоссе",
                "пересказывают"
            };
            
            var results = m.Parse(words).ToArray();

            foreach (var morphInfo in results)
            {
                if (morphInfo.HasLemma("королевский"))    
                    Console.WriteLine(morphInfo.Text);
            }
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