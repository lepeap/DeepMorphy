using System;
using NUnit.Framework;

namespace DeepMorphy.Tests
{
    public class TagHelperTest
    {
        public MorphAnalyzer MorphRu { get; set; }

        public MorphAnalyzer MorphEn { get; set; }

        [SetUp]
        public void Setup()
        {
            MorphRu = new MorphAnalyzer();
            MorphEn = new MorphAnalyzer(useEnGrams: true);
        }

        [Test]
        public void InfnTag()
        {
            var word = "тестить";
            var enTag = MorphEn.TagHelper.CreateForInfn(word);
            var ruTag = MorphRu.TagHelper.CreateForInfn(word);
            CheckTag(MorphEn, enTag, post: "infn", word);
            CheckTag(MorphRu, ruTag, post: "infn", word);
        }
        
        [Test]
        public void NounTag1()
        {
            var word = "стол";
            var post = "noun";
            var gndr = "masc";
            var nmbr = "sing";
            var @case = "nomn";

            var enTag = MorphEn.TagHelper.CreateForNoun(word,
                number: nmbr,
                gender: gndr,
                @case: @case);
            
            var ruTag = MorphRu.TagHelper.CreateForNoun(word,
                number: _t(nmbr),
                gender: _t(gndr),
                @case: _t(@case));

            CheckTag(
                MorphEn,
                enTag,
                post: post, 
                gndr: gndr,
                nmbr: nmbr,
                @case: @case,
                lemma: word,
                adMessageText: "(вызов без заданной леммы)");
            CheckTag(
                MorphRu,
                ruTag,
                post: post, 
                gndr: gndr,
                nmbr: nmbr,
                @case: @case,
                lemma: word,
                adMessageText: "(вызов без заданной леммы)");
        }

        private void CheckTag(MorphAnalyzer morph,
            Tag tag,
            string post,
            string lemma=null,
            string gndr=null,
            string nmbr=null,
            string @case=null,
            string tens=null,
            string pers=null,
            string adMessageText="")
        {
            if (!morph.EnTags)
            {
                post = morph.GramHelper.TranslateKeyToRu(post);
                gndr = gndr != null ? morph.GramHelper.TranslateKeyToRu(gndr) : null;
                nmbr = nmbr != null ? morph.GramHelper.TranslateKeyToRu(nmbr) : null;
                @case = @case != null ? morph.GramHelper.TranslateKeyToRu(@case) : null;
                tens = tens != null ? morph.GramHelper.TranslateKeyToRu(tens) : null;
                pers = pers != null ? morph.GramHelper.TranslateKeyToRu(pers) : null;
            }

            Assert.AreEqual(post, tag.Post, $"Неправильная часть речи {adMessageText}");
            Assert.AreEqual(gndr, tag.Gender, $"Неправильный род {adMessageText}");
            Assert.AreEqual(nmbr, tag.Number, $"Неправильное число {adMessageText}");
            Assert.AreEqual(@case, tag.Case, $"Неправильный часть речи {adMessageText}");
            Assert.AreEqual(tens, tag.Tens, $"Неправильное время {adMessageText}");
            Assert.AreEqual(pers, tag.Pers, $"Неправильное лицо {adMessageText}");
            Assert.AreEqual(lemma, tag.Lemma, $"Неправильная лемма {adMessageText}");
        }

        private string _t(string key)
        {
            return MorphRu.GramHelper.TranslateKeyToRu(key);
        }
    }
}