using System;
using System.IO;
using DeepMorphy;
using DeepMorphy.WordDict;

namespace IntegrationTester
{
    /// <summary>
    /// Metrics calculator for Neural Net (only NN without dictionary and preprocessors)
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            using (var filestream = new FileStream("log.txt", FileMode.Create))
            {
                using (var streamwriter = new StreamWriter(filestream))
                {
                    streamwriter.AutoFlush = true;
                    Console.SetOut(streamwriter);
                    Console.SetError(streamwriter);
                    Console.WriteLine(DateTime.Now);
                    
                    var morph = new MorphAnalyzer(useEnGramNames: true, withLemmatization: true);
                    morph.CorrectionDict = new Dict();
                    morph.Processors = new IMorphProcessor[0];
                    new Tester(morph,  "Network", "Only network").Test();
                    morph = new MorphAnalyzer(useEnGramNames: true, withLemmatization: true);
                    new Tester(morph, "Network", "Full").Test();
                    new Tester(morph, "Reg", "Reg").Test();
                    new Tester(morph, "Numb", "Numb").Test();
                    new Tester(morph, "NarNumb", "NarNumb").Test();
                    new Tester(morph, "Dict", "Dict").Test();
                }
            }
        }
    }
}