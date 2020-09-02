using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml;
using DeepMorphy;

namespace MetricsCalc
{
    public class TestsCalc
    {
        private readonly string _pathPrefix;
        private readonly string _testName;
        private readonly MorphAnalyzer _morph;

        internal TestsCalc(MorphAnalyzer morph, string pathPrefix, string testName)
        {
            _morph = morph;
            _pathPrefix = pathPrefix;
            _testName = testName;
        }

        internal void Test()
        {
            TestGramClassification();
            TestMainClassification();
            TestLemmatization();
            TestInflect();
        }

        private void TestLemmatization()
        {
            var tests = LoadTests("lemma").ToArray();
            float totalCount = tests.Length;
            float correctCount = 0;
            int i = 0;
            var tasks = tests.Select(x => (x.X, GetTag(x.ClsX)));
            var results = _morph.Lemmatize(tasks).ToArray();
            foreach (var res in results)
            {
                var test = tests[i];
                if (res == test.Y)
                {
                    correctCount++;
                }
                else
                {
                    Console.WriteLine($"{_testName} lemmatization. Error: {test.Y} != {res}");
                }

                i++;
            }

            float result = correctCount / totalCount;
            Console.WriteLine($"{_testName} lemmatization acc: {result}");
            Console.WriteLine($"{_testName} lemmatization errors count: {totalCount - correctCount}");
        }

        private void TestInflect()
        {
            var tests = LoadTests("inflect").ToArray();
            float totalCount = tests.Length;
            float correctCount = 0;
            int i = 0;
            var tasks = tests.Select(x => (x.X, GetTag(x.ClsX), GetTag(x.ClsY)));

            foreach (var res in _morph.Inflect(tasks))
            {
                var test = tests[i];
                if (res == test.Y)
                {
                    correctCount++;
                }
                else
                {
                    var tagText = string.Join(",", TagHelper.TagsRuDic[test.ClsY].Select(kp => kp.Value));
                    Console.WriteLine($"{_testName} inflect. Error: {test.Y} != {res}. Result tag: {tagText}");
                }
                i++;
            }

            float result = correctCount / totalCount;
            Console.WriteLine($"{_testName} inflect acc: {result}");
            Console.WriteLine($"{_testName} inflect errors count: {totalCount - correctCount}");
        }

        private void TestGramClassification()
        {
            var grams = Directory.GetFiles(_pathPrefix, "*.xml")
                .Select(Path.GetFileNameWithoutExtension)
                .Where(x => x != "lemma" && x != "main" && x != "inflect")
                .ToArray();

            foreach (var gram in grams)
            {
                var tests = LoadTests(gram).ToArray();
                var results = _morph.Parse(tests.Select(x => x.X)).ToArray();
                float testsCount = tests.Length;
                float totalClassesCount = 0;
                float correctTests = 0;
                float correctClassesCount = 0;

                for (int i = 0; i < tests.Length; i++)
                {
                    var test = tests[i];
                    var res = results[i];
                    var etRez = test.Y.Split(';');

                    var rez = res[gram].Grams.ToArray();
                    totalClassesCount += etRez.Length;

                    bool correct = true;
                    for (int j = 0; j < etRez.Length; j++)
                    {
                        if (etRez.Contains(rez[j].Key))
                        {
                            correctClassesCount++;
                        }
                        else
                        {
                            correct = false;
                            break;
                        }
                    }

                    if (correct)
                    {
                        correctTests++;
                    }
                }

                float testAcc = correctTests / testsCount;
                float clsAcc = correctClassesCount / totalClassesCount;
                Console.WriteLine($"{_testName} {gram} classification. Full acc: {testAcc}");
                Console.WriteLine($"{_testName} {gram} classification. Classes acc: {clsAcc}");
            }
        }

        private void TestMainClassification()
        {
            var tests = LoadTests("main").ToArray();
            var results = _morph.Parse(tests.Select(x => x.X)).ToArray();
            float testsCount = tests.Length;
            float totalClassesCount = 0;
            float correctTests = 0;
            float correctClassesCount = 0;
            for (int i = 0; i < tests.Length; i++)
            {
                var test = tests[i];
                var res = results[i];
                var etalonRez = test.Y.Split(';').Select(x => int.Parse(x)).ToArray();
                totalClassesCount += etalonRez.Length;
                int curCount = 0;
                bool error = false;
                foreach (var etIndex in etalonRez)
                {
                    if (res.Tags.Any(t => t.Id == etIndex))
                    {
                        curCount++;
                    }
                    else
                    {
                        error = true;
                    }
                }

                if (error)
                {
                    Console.WriteLine($"{_testName} main. Error: {test.X}");
                }

                correctClassesCount += curCount;
                if (curCount == etalonRez.Length)
                {
                    correctTests++;
                }
            }

            float testAcc = correctTests / testsCount;
            float clsAcc = correctClassesCount / totalClassesCount;
            Console.WriteLine($"{_testName} main classification. Full acc: {testAcc}");
            Console.WriteLine($"{_testName} main classification. Classes acc: {clsAcc}");
            Console.WriteLine($"{_testName} main errors count: {totalClassesCount - correctClassesCount}");
        }

        private IEnumerable<Test> LoadTests(string name)
        {
            var path = Path.Combine(_pathPrefix, $"{name}.xml");
            using (Stream stream = File.Open(path, FileMode.Open))
            {
                var rdr = XmlReader.Create(new StreamReader(stream, Encoding.UTF8));
                while (rdr.Read())
                {
                    if (rdr.Name == "T" && rdr.NodeType == XmlNodeType.Element)
                    {
                        yield return new Test()
                        {
                            X = rdr.GetAttribute("x"),
                            ClsX = rdr.GetAttribute("x_c") != null ? int.Parse(rdr.GetAttribute("x_c")) : -1,
                            Y = rdr.GetAttribute("y"),
                            ClsY = rdr.GetAttribute("y_c") != null ? int.Parse(rdr.GetAttribute("y_c")) : -1
                        };
                    }
                }
            }
        }

        private Tag GetTag(int id)
        {
            return _morph.TagHelper.CreateTagFromId(id);
        }
    }
}