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
        static void Main(string[] args)
        {
            //ShowMemoryInfo();
            var morph = new MorphAnalyzer(useEnGramNames: true, onlyNetwork: true, withLemmatization: true);
            new MetricsCalc(morph,  "NnTests", "Only network").CalcMetrics();
            morph = new MorphAnalyzer(useEnGramNames: true, onlyNetwork: false, withLemmatization: true);
            new MetricsCalc(morph, "NnTests", "Full").CalcMetrics();
        }

        private static long GetMemory()
        {
            return GC.GetTotalMemory(false);
        }

        private static void ShowMemoryInfo()
        {
            Console.WriteLine("Memory consumption info");            
            Console.WriteLine($"Before all: {GetMemory()}");
            var morph = new MorphAnalyzer(onlyNetwork: true, withLemmatization: true);
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
    }
}