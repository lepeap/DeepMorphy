using DeepMorphy;
using NUnit.Framework;

namespace UnitTests
{
    public class TagHelperTest
    {
        [Test]
        public void TestExistingEnTagCreation()
        {
            _testExistingTagCreation(new MorphAnalyzer(useEnGramNames: true));
        }

        [Test]
        public void TestExistingRuTagCreation()
        {
            _testExistingTagCreation(new MorphAnalyzer());
        }

        private void _testExistingTagCreation(MorphAnalyzer morph)
        {
            string postKey = "post";
            string nmbrKey = "nmbr";
            string gndrKey = "gndr";
            string caseKey = "case";
            string persKey = "pers";
            string tensKey = "tens";
            string moodKey = "mood";
            string voicKey = "voic";
            if (!morph.UseEnGramNameNames)
            {
                postKey = GramInfo.TranslateKeyToRu(postKey);
                nmbrKey = GramInfo.TranslateKeyToRu(nmbrKey);
                gndrKey = GramInfo.TranslateKeyToRu(gndrKey);
                caseKey = GramInfo.TranslateKeyToRu(caseKey);
                persKey = GramInfo.TranslateKeyToRu(persKey);
                tensKey = GramInfo.TranslateKeyToRu(tensKey);
                moodKey = GramInfo.TranslateKeyToRu(moodKey);
                voicKey = GramInfo.TranslateKeyToRu(voicKey);
            }

            foreach (var kp in morph.TagHelper.TagsDic)
            {
                var index = kp.Key;
                var gDic = kp.Value;
                var post = gDic[postKey];
                var nmbr = gDic.ContainsKey(nmbrKey) ? gDic[nmbrKey] : null;
                var gndr = gDic.ContainsKey(gndrKey) ? gDic[gndrKey] : null;
                var @case = gDic.ContainsKey(caseKey) ? gDic[caseKey] : null;
                var pers = gDic.ContainsKey(persKey) ? gDic[persKey] : null;
                var tens = gDic.ContainsKey(tensKey) ? gDic[tensKey] : null;
                var mood = gDic.ContainsKey(moodKey) ? gDic[moodKey] : null;
                var voic = gDic.ContainsKey(voicKey) ? gDic[voicKey] : null;

                var tagWithoutLemma = morph.TagHelper.CreateTag(
                    post,
                    gndr = gndr,
                    nmbr = nmbr,
                    @case = @case,
                    pers = pers,
                    tens = tens,
                    mood = mood,
                    voic = voic);

                var tagWithLemma = morph.TagHelper.CreateTag(
                    post,
                    gndr = gndr,
                    nmbr = nmbr,
                    @case = @case,
                    pers = pers,
                    tens = tens,
                    mood = mood,
                    voic = voic,
                    lemma: "111");

                Assert.AreEqual(index, tagWithoutLemma.Id, "Неправильный айди при создании без леммы");
                Assert.AreEqual(index, tagWithoutLemma.Id, "Неправильный айди при создании c леммой");
                Assert.IsNull(tagWithoutLemma.Lemma, "Лемма заполнена (должна быть пустой)");
                Assert.AreEqual("111", tagWithLemma.Lemma, "Неправильная лемма");
            }
        }
    }
}