using System.Linq;
using DeepMorphy;
using NUnit.Framework;

namespace UnitTests
{
    public class LongestCommonPrefixTest
    {
        [Test]
        public void LcpSimple1()
        {
            var ar = new[]
            {
                "111",
                "111222",
                "111222333"
            };

            var lcp = ar.GetLongestCommonPrefix();
            Assert.AreEqual("111", lcp);
        }
        
        [Test]
        public void NoLcp()
        {
            var ar = new string[]
            {
                "111",
                "222"
            };

            var lcp = ar.GetLongestCommonPrefix();
            Assert.AreEqual(string.Empty, lcp);
        }
                
        [Test]
        public void LcpEmptyInputArray()
        {
            var ar = new string[0];
            var lcp = ar.GetLongestCommonPrefix();
            Assert.AreEqual(string.Empty, lcp);
        }
        
        [Test]
        public void Lcp1Char()
        {
            var ar = new[]
            {
                "3111",
                "3222"
            };
            
            var lcp = ar.GetLongestCommonPrefix();
            Assert.AreEqual("3", lcp);
        }
        
        [Test]
        public void LcpSimple1WithEndings()
        {
            var ar = new[]
            {
                "111",
                "111222",
                "111222333"
            };

            var lcp = ar.GetLongestCommonPrefixWithEndings(out string[] endings);
            var endingsEtalon = new[]
            {
                "222",
                "222333"
            };
            
            Assert.AreEqual("111", lcp);
            Assert.AreEqual(endingsEtalon, endings);
        }
        
        [Test]
        public void NoLcpWithEndings()
        {
            var ar = new string[]
            {
                "111",
                "222"
            };

            var lcp = ar.GetLongestCommonPrefixWithEndings(out string[] endings);
            Assert.AreEqual(string.Empty, lcp);
            Assert.AreEqual(0, endings.Length);
        }
        
        [Test]
        public void LcpEmptyInputArrayWithEndings()
        {
            var ar = new string[0];
            var lcp = ar.GetLongestCommonPrefixWithEndings(out string[] endings);
            Assert.AreEqual(string.Empty, lcp);
            Assert.AreEqual(0, endings.Length);
        }
        
        [Test]
        public void Lcp1CharWithEndings()
        {
            var ar = new[]
            {
                "3111",
                "3222"
            };
            
            var lcp = ar.GetLongestCommonPrefixWithEndings(out string[] endings);
            var etalonEndings = new[]
            {
                "111",
                "222"
            };
            
            Assert.AreEqual("3", lcp);
            Assert.AreEqual(etalonEndings, endings);
        }

        [Test]
        public void StemTest()
        {
            var morph = new MorphAnalyzer();
            var lexeme = morph.Lexeme("федеральный", morph.TagHelper.CreateTag("прил", gndr: "муж", nmbr: "ед", @case: "им"));
            var prils = lexeme.Where(x => x.tag.Has("прил"));
            var stem = prils.GetLongestCommonPrefixWithEndings(out string[] endings);
            
            Assert.AreEqual("федеральн", stem);

            var etalonEndings = new[]
            {
                "ых",
                "ыми", 
                "ым",
                "ые",
                "ом", 
                "ое", 
                "ому", 
                "ого", 
                "ой", 
                "ую",
                "ая",
                "ый"
            };
            Assert.AreEqual(etalonEndings, endings);
        }
    }
}