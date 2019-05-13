using System;
using System.Linq;
using DeepMorphy;

namespace ExampleConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            var m = new MorphAnalyzer(withLemmatization: true, withTrimAndLower: false);
            var results = m.Parse(new string[]
            {
                "tafsdfdfasd",
                "xii",
                "123",
                ".345",
                "43,34",
                "..!",
                "а",
                "еж",
                "хуй",
                "хороший",
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
                "центральные"
                
            }).ToArray();

            foreach (var token in results)
                Console.WriteLine(token.ToString());
        }
    }
}