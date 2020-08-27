using System.Linq;
using DeepMorphy.Numb;
using NUnit.Framework;

namespace DeepMorphy.Tests
{
    public class NarNumberTest
    {
        internal IMorphProcessor proc;
        
        [SetUp]
        public void Setup()
        {
            //proc = new NumberProc(false, true);
        }

        [Test]
        public void SimpleCls1()
        {
            //var results = _parse("двухтысячной");
            //results = _parse("ноля");
            //results = _parse("пятьсотпервого");
            //var lexeme = proc.Lexeme(results[0].Text, results[0].BestTag).ToArray();
            //var result = proc.Inflect(new[]
            //{
            //    (results[0].Text, results[0].BestTag, lexeme[3].tag)
            //});

            //var results = _parse("1-й");
            //results = _parse("101-ой");
            //results = _parse("1917-ый");
            //var result = narProc.Inflect(new[]
            //{
            //    (results[0].Text, results[0].BestTag, lexeme[3].tag)
            //})
            
            
        }


        
        //private MorphInfo[] _parse(params string[] words)
        //{
        //    //return proc.Parse(words).ToArray();
        //}
    }
}