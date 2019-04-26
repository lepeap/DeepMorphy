using System.IO;
using System.Reflection;

namespace DeepMorphy
{
    static class Utils
    {
        private static Assembly _assembly = Assembly.GetExecutingAssembly();
        public static Stream GetResourceStream(string name)
        {
            return _assembly.GetManifestResourceStream(name);
        }
        
    }
}