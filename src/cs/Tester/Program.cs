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
            using (var filestream = new FileStream("log.txt", FileMode.Create))
            {
                using (var streamwriter = new StreamWriter(filestream))
                {
                    streamwriter.AutoFlush = true;
                    Console.SetOut(streamwriter);
                    Console.SetError(streamwriter);
                    Console.WriteLine(DateTime.Now);
                    
                    var morph = new MorphAnalyzer(useEnGramNames: true, onlyNetwork: true, withLemmatization: true);
                    new TestsCalc(morph,  "Network", "Only network").Test();
                    morph = new MorphAnalyzer(useEnGramNames: true, onlyNetwork: false, withLemmatization: true);
                    new TestsCalc(morph, "Network", "Full").Test();
                    new TestsCalc(morph, "Reg", "Reg").Test();
                    new TestsCalc(morph, "Numb", "Numb").Test();
                    new TestsCalc(morph, "NarNumb", "NarNumb").Test();
                    new TestsCalc(morph, "Dict", "Dict").Test();
                }
            }
        }
    }
}