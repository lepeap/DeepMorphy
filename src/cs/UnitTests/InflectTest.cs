using System.Linq;
using DeepMorphy;
using DeepMorphy.Model;
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
        
        [Test]
        public void UnsupportedNetworkInflect()
        {
            var word = "тест";
            var srcTag = _morph.TagHelper.CreateTag("прил", gndr: "жен", nmbr: "ед", @case: "им");
            var resTag = _morph.TagHelper.CreateTag("предл");
            var inflectTask = new []
            {
                _t(word, srcTag, resTag)
            };
            var result = _morph.Inflect(inflectTask).First();
            Assert.IsNull(result);
        }
        
        [Test]
        public void UnsupportedNumbInflect()
        {
            var word = "тест";
            var srcTag = _morph.TagHelper.CreateTag("числ", gndr: "муж", nmbr: "ед", @case: "им");
            var resTag = _morph.TagHelper.CreateTag("предл");
            var inflectTask = new []
            {
                _t(word, srcTag, resTag)
            };
            var result = _morph.Inflect(inflectTask).First();
            Assert.IsNull(result);
        }
        
        [Test]
        public void UnsupportedDictInflect()
        {
            var word = "тест";
            var srcTag = _morph.TagHelper.CreateTag("союз");
            var resTag = _morph.TagHelper.CreateTag("предл");
            var inflectTask = new []
            {
                _t(word, srcTag, resTag)
            };
            var result = _morph.Inflect(inflectTask).First();
            Assert.IsNull(result);
        }

        private InflectTask _t(string word, Tag srcTag, Tag resTag)
        {
            return new InflectTask(word, srcTag, resTag);
        }
    }
}