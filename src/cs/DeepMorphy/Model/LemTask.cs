namespace DeepMorphy.Model
{
    /// <summary>
    /// Задание на лемматизацию
    /// </summary>
    public class LemTask
    {
        public LemTask(string word, Tag tag)
        {
            this.word = word;
            this.tag = tag;
        }

        /// <summary>
        /// Слово, которое нужно лемматизировать
        /// </summary>
        public readonly string word;

        /// <summary>
        /// Тэг слова
        /// </summary>
        public readonly Tag tag;
    }
}