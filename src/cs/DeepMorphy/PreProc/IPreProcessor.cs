namespace DeepMorphy.PreProc
{
    interface IPreProcessor
    {
        Token Parse(string word);
    }
}