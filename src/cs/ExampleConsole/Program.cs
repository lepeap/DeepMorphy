using System;
using System.Linq;
using DeepMorphy;

namespace ExampleConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            var m = new MorphAnalyzer(withTrimAndLower: false);
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
                "ебанат",
                "хуй",
                "ебаться",
                "хороший",
                "1-ый",
                "бутявка",
                "в",
                "действуя",
                "пизданат",
                "пизда",
                "двадцать",
                "тысячу",
                "миллионных",
                "222-ого",
                "дотошный",
                "красотка"
            }).ToArray();

            foreach (var token in results)
                Console.WriteLine(token.ToString());
        }
    }
}