using System.Linq;
using DeepMorphy;
using NUnit.Framework;

namespace UnitTests
{
    public class ParseTest
    {
        private MorphAnalyzer _morph;
        
        [SetUp]
        public void Setup()
        {
            _morph = new MorphAnalyzer();
        }
        
        [Test]
        public void SimplePorNumbSameLem()
        {
            var word = "тысячный";
            var morf = _morph.Parse(word).First();
            
            Assert.AreEqual("числ", morf.BestTag["чр"]);
            Assert.AreEqual("муж", morf.BestTag["род"]);
            Assert.AreEqual("ед", morf.BestTag["число"]);
            Assert.AreEqual("им", morf.BestTag["падеж"]);
            Assert.AreEqual(word, morf.BestTag.Lemma);
        }
        
                
        [Test]
        public void CombinedPorNumbSameLem()
        {
            var word = "двадцатипятитысячный";
            var morf = _morph.Parse(word).First();
            
            Assert.AreEqual("числ", morf.BestTag["чр"]);
            Assert.AreEqual("муж", morf.BestTag["род"]);
            Assert.AreEqual("ед", morf.BestTag["число"]);
            Assert.AreEqual("им", morf.BestTag["падеж"]);
            
            Assert.AreEqual(word, morf.BestTag.Lemma);
        }
    }
}