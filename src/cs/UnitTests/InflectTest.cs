using System.Linq;
using DeepMorphy;
using NUnit.Framework;

namespace UnitTests
{
    public class InflectTest
    {
        private MorphAnalyzer _morph;
        
        [SetUp]
        public void Setup()
        {
            _morph = new MorphAnalyzer();
        }
        
        [Test]
        public void SimpleInflect()
        {
            var word = "большой";
            var srcTag = _morph.TagHelper.CreateTag("прил", gndr: "муж", nmbr: "ед", @case: "им");
            var resTag = _morph.TagHelper.CreateTag("прил", nmbr: "мн", @case: "им");
            var inflectTask = new []
            {
                _t(word, srcTag, resTag)
            };
            var result = _morph.Inflect(inflectTask).First();
            Assert.AreEqual("большие", result);
        }

        [Test]
        public void NeedLemmatizeInflect()
        {
            var word = "большая";
            var srcTag = _morph.TagHelper.CreateTag("прил", gndr: "жен", nmbr: "ед", @case: "им");
            var resTag = _morph.TagHelper.CreateTag("прил", nmbr: "мн", @case: "им");
            var inflectTask = new []
            {
                _t(word, srcTag, resTag)
            };
            var result = _morph.Inflect(inflectTask).First();
            Assert.AreEqual("большие", result);
        }

        private (string word, Tag wordTag, Tag resultTag) _t(string word, Tag srcTag, Tag resTag)
        {
            return (word: word, wordTag: srcTag, resultTag: resTag);
        }
    }
}