using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using DeepMorphy;
using BenchmarkDotNet.Attributes;

namespace Benchmark
{
    [MemoryDiagnoser]
    public class MainBench
    {
        private static readonly string[] Words = new[]
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
        };
        private static MorphAnalyzer Morph = new MorphAnalyzer();

        private static MorphInfo[] _process(string[] words)
        {
            return Morph.Parse(words).ToArray();
        }

        private static string[] _repeatWords(int n)
        {
            var list = new List<string>();
            for (int i = 0; i < n; i++)
            {
                list.AddRange(Words);
            }

            return list.ToArray();

        }
         
        [Benchmark]
        [ArgumentsSource(nameof(GetData))]
        public MorphInfo[] Process(string[] words, int value)
        {
            return _process(words);
        }
        
        public IEnumerable<object[]> GetData()
        {
            yield return new object[] { Words, Words.Length };
            
            var words = _repeatWords(10);
            yield return new object[] { words, words.Length };

            words = _repeatWords(100);
            yield return new object[] { words, words.Length };
            
            words = _repeatWords(1000);
            yield return new object[] { words, words.Length };
        }
    }
}