using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace DataReader
{
    class Program
    {
        static void Main(string[] args)
        {
            string filePath = Path.Combine(Environment.CurrentDirectory, @"trainingdata.csv");      // file that has the training data
            int LINESIZE = 5250;                                                                    // how many characters to read from a line
            int NUM_ENTRIES = 105472;                                                               // number of entries in the excel sheet
            int NUM_FEATURES = 772;                                                                 // this number is the loss
            string userInput;                                                                       // user supplied input to convert to feature number
            int feature;                                                                            // which feature does the user want?
            double[] data = new double[NUM_ENTRIES];                                                // array to hold the feature data
            double[] loanLoss = new double[NUM_ENTRIES];                                            // parallel array to data that holds the loan loss information
            double min = 0, max = 0;                                                                // min and max for the reange of the feature
            string[] temporaryArray = new string[1];                                                // temporary variable for the line in the loop

            Console.WriteLine("Which feature would you like to analyze?");      // prompting user for input
            userInput = Console.ReadLine();                                     // reading user input
            feature = Convert.ToInt32(userInput);

            Console.WriteLine(feature);

            using (FileStream fs = File.Open(filePath, FileMode.Open, FileAccess.Read, FileShare.None))
            {
                byte[] b = new byte[LINESIZE];                  // buffer to read from the file
                UTF8Encoding temp = new UTF8Encoding(true);     // I'm not 100% sure what this is

                while (fs.Read(b, 0, b.Length) > 0)            // reading a line
                {
                    temporaryArray = System.Text.Encoding.UTF8.GetString(b).Split(',');
                }
            }
        }
    }
}
