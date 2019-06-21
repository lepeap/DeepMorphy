using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks.Dataflow;
using System.Xml;
using DeepMorphy;

namespace MetricsCalc
{
    /// <summary>
    /// Metrics calculator for Neural Net (only NN without dictionary and preprocessors)
    /// </summary>
    class Program
    {
        class Test
        {
            public string X { get; set; }
            public string Y { get; set; }
        }
        static void Main(string[] args)
        {
            ShowMemoryInfo();
            TestMainClassification();
            TestLemmatization();
            TestGramClassification();

        }

        private static long GetMemory()
        {
            return GC.GetTotalMemory(false);
        }

        private static void ShowMemoryInfo()
        {
            Console.WriteLine("Memory consumption info");            
            Console.WriteLine($"Before all: {GetMemory()}");
            var morph = new MorphAnalyzer(withPreprocessors: true, withLemmatization: true);
            Console.WriteLine($"After init: {GetMemory()}");
            int j = 0;
            while (j < 100)
            {
                var results = morph.Parse(new string[]
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
                    "укрывал",
                    "королевские",
                    "корабли",
                    "укрывал",
                    "обновляя",
                    "выходящие",
                    "собаковод",
                    "раскладывала", 
                    "обучает",
                    "юбка",
                    "пересказывают"
                }).ToArray();
                j++;
            }
            Console.WriteLine($"After processing: {GetMemory()}");
            GC.Collect();
            Console.WriteLine($"After collect: {GetMemory()}");
        }

        private static void TestLemmatization()
        {
            Console.WriteLine("Calculating lemmatization");
            var morph = new MorphAnalyzer(useEnGrams: true, withPreprocessors: true, withLemmatization: true);
            var tests = LoadTests("lem").ToArray();
            float totalCount = tests.Length;
            float correctCount = 0;
            int i = 0;
            foreach (var res in  morph.Parse(tests.Select(x => x.X)))
            {
                var test = tests[i];
                if (res.HasLemma(test.Y))
                    correctCount++;
                
                i++;
            }
            float result = correctCount / totalCount;
            Console.WriteLine($"Lemmatization acc: {result}");
        }
        
        private static void TestGramClassification()
        {
            var grams = Directory.GetFiles(System.Environment.CurrentDirectory, "*.xml")
                .Select(Path.GetFileNameWithoutExtension)
                .Where(x => x != "lem" && x != "main")
                .ToArray();

            foreach (var gram in grams)
            {
                Console.WriteLine($"Calculating {gram} classification");
                var morph = new MorphAnalyzer(useEnGrams: true, withPreprocessors: true);
                var tests = LoadTests(gram).ToArray();
                var results = morph.Parse(tests.Select(x => x.X)).ToArray();
                float testsCount = tests.Length;
                float totalClassesCount = 0;
                float correctTests = 0;
                float correctClassesCount = 0;
                
                for (int i = 0; i < tests.Length; i++)
                {
                    var test = tests[i];
                    var res = results[i];
                    var etRez = test.Y.Split(';');
                    
                    var rez = res[gram].Grams.ToArray();
                    totalClassesCount += etRez.Length;

                    bool correct = true;
                    for (int j = 0; j < etRez.Length; j++)
                    {
                        if (etRez.Contains(rez[j].Key))
                            correctClassesCount++;
                        else
                        {
                            correct = false;
                            break;
                        }
                    }

                    if (correct)
                        correctTests++;
                    
                }

                float testAcc = correctTests / testsCount;
                float clsAcc = correctClassesCount / totalClassesCount;
                Console.WriteLine($"{gram} classification. Full acc: {testAcc}");
                Console.WriteLine($"{gram} classification. Classes acc: {clsAcc}");
                
            }
        }
        
        private static void TestMainClassification()
        {
            Console.WriteLine("Calculating main classification");
            var morph = new MorphAnalyzer(useEnGrams: true, withPreprocessors: true);
            var tests = LoadTests("main").ToArray();
            var results = morph.Parse(tests.Select(x => x.X)).ToArray();
            float testsCount = tests.Length;
            float totalClassesCount = 0;
            float correctTests = 0;
            float correctClassesCount = 0;
            for (int i = 0; i < tests.Length; i++)
            {
                var test = tests[i];
                var res = results[i];
                var testRez = test.Y.Split(';');
                totalClassesCount += testRez.Length;
                int curCount = 0;
                foreach (var tag in res.Tags)
                {
                    if (testRez.Contains(tag.ToString()))
                        curCount++;
                }
                correctClassesCount += curCount;
                if (curCount == testRez.Length)
                    correctTests++;
            }

            float testAcc = correctTests / testsCount;
            float clsAcc = correctClassesCount / totalClassesCount;
            Console.WriteLine($"Main classification. Full acc: {testAcc}");
            Console.WriteLine($"Main classification. Classes acc: {clsAcc}");
        }

        private static IEnumerable<Test> LoadTests(string name)
        {
            using (Stream stream = File.Open($"{name}.xml", FileMode.Open))
            {
                var rdr = XmlReader.Create(new StreamReader(stream, Encoding.UTF8));
                while (rdr.Read())
                {

                    if (rdr.Name == "T" && rdr.NodeType == XmlNodeType.Element)
                    {
                        yield return new Test()
                        {
                            X = rdr.GetAttribute("x"),
                            Y = rdr.GetAttribute("y")
                        };
                    }
                }
            }
        }
    }
}