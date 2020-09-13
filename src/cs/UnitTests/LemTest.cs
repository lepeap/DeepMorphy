using System.Linq;
using DeepMorphy;
using DeepMorphy.Model;
using NUnit.Framework;

namespace UnitTests
{
    public class LemTest
    {
        private MorphAnalyzer _morph;
        
        [SetUp]
        public void Setup()
        {
            _morph = new MorphAnalyzer();
        }
        
        [Test]
        public void NarNumbTest1()
        {
            var word = "10-ю";
            var tag = _morph.TagHelper.CreateTag("числ", gndr: "жен", nmbr: "ед", @case: "дт");
            var task = new []
            {
                new LemTask(word, tag)
            };
            var result = _morph.Lemmatize(task).First();
            Assert.AreEqual("10-й", result);
        }
    }
}