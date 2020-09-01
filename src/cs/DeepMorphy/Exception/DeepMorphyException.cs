namespace DeepMorphy.Exception
{
    /// <summary>
    /// Базовое исключение библиотеки
    /// </summary>
    public class DeepMorphyException : System.Exception
    {
        public DeepMorphyException(string message) : base(message)
        {
            
        }
    }
}