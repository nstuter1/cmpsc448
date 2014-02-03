using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Text;


namespace DataReader
{
    class Program
    {
        static void Main(string[] args)
        {
            string filePath = Path.Combine(Environment.CurrentDirectory, @"trainingdata.csv");      // file that has the training data
            int LINESIZE = 5250;                                                                    // how many characters to read from a line
            string userInput;                                                                       // user supplied input to convert to feature number
            int feature;                                                                            // which feature does the user want?

            Console.WriteLine("Which feature would you like to analyze?");      // prompting user for input
            userInput = Console.ReadLine();                                     // reading user input
            feature = Convert.ToInt32(userInput);

            Console.WriteLine(feature);

            using (FileStream fs = File.Open(filePath, FileMode.Open, FileAccess.Read, FileShare.None))
            {
                byte[] b = new byte[LINESIZE];                  // buffer to read from the file
                UTF8Encoding temp = new UTF8Encoding(true);     // I'm not 100% sure what this is

                while( fs.Read(b, 0, b.Length) > 0 )            // reading a line
                {
                    Console.WriteLine(temp.GetString(b));       // printing the line read
                }
            }
        }
    }
}
