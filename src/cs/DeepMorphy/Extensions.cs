using System.Collections.Generic;
using System.Linq;
using DeepMorphy.Model;

namespace DeepMorphy
{
    public static class Extensions
    {
        /// <summary>
        /// Возвращает наибольших общий префикс для перечисления слов
        /// </summary>
        /// <param name="words">Перечисление слов</param>
        /// <returns>Наибольших общий префикс</returns>
        public static string GetLongestCommonPrefix(this IEnumerable<string> words)
        {
            var wordsAr = words.ToArray();
            return wordsAr.GetLongestCommonPrefix();
        }

        /// <summary>
        /// Возвращает наибольших общий префикс для перечисления слов
        /// </summary>
        /// <param name="words">Перечисление слов</param>
        /// <returns>Наибольших общий префикс</returns>
        public static string GetLongestCommonPrefix(this IEnumerable<(Tag tag, string text)> words)
        {
            return words.Select(x => x.text).GetLongestCommonPrefix();
        }

        /// <summary>
        /// Возвращает наибольших общий префикс для перечисления слов + различающиеся окончания
        /// </summary>
        /// <param name="words">Перечисление слов</param>
        /// <param name="endings">Различающиеся окончания</param>
        /// <returns>Наибольших общий префикс</returns>
        public static string GetLongestCommonPrefixWithEndings(this IEnumerable<string> words, out string[] endings)
        {
            var wordsAr = words.ToArray();
            var commonPrefix = wordsAr.GetLongestCommonPrefix();
            endings = string.IsNullOrEmpty(commonPrefix) 
                    ? new string[0]
                    : wordsAr.Select(x => x.Remove(0, commonPrefix.Length))
                             .Where(x => !string.IsNullOrWhiteSpace(x))
                             .Distinct()
                             .ToArray();
            return commonPrefix;
        }

        /// <summary>
        /// Возвращает наибольших общий префикс для перечисления слов + различающиеся окончания
        /// </summary>
        /// <param name="words">Перечисление слов</param>
        /// <param name="endings">Различающиеся окончания</param>
        /// <returns>Наибольших общий префикс</returns>
        public static string GetLongestCommonPrefixWithEndings(this IEnumerable<(Tag tag, string text)> words,
            out string[] endings)
        {
            var wordsEn = words.Select(x => x.text);
            return wordsEn.GetLongestCommonPrefixWithEndings(out endings);
        }

        internal static string GetLongestCommonPrefix(this string[] words)
        {
            if (words.Length == 0)
            {
                return string.Empty;
            }

            var minLength = words.Min(x => x.Length);
            int i = minLength - 1;
            while (i >= 0 && words.Any(x => x[i] != words[0][i]))
            {
                i--;
            }

            if (i < 0)
            {
                return string.Empty;
            }

            return words[0].Substring(0, i + 1);
        }
    }
}