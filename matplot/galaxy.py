import argparse
import common
import correlation_func
import dicer
import stats
import survey

def main():
    #Handle command line arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    #Generate settings file
    parser_gensettings = subparsers.add_parser('sset', help="generates a sample settings file")
    parser_gensettings.add_argument("-m","--module",type=str,default="correlation",help="Which module to generate settings for. Default: correlation.")
    parser_gensettings.set_defaults(func=common.gensettings)

    
    parser_correlation = subparsers.add_parser('correlation', help="runs the correlation function")
    parser_correlation.add_argument("settings",help="Read in settings from this file.", type=str)
    parser_correlation.set_defaults(func=correlation_func.mainrun)

    parser_breakbox = subparsers.add_parser('divide', help='runs the "dicer" function to divide a huge box into smaller boxes')
    parser_breakbox.add_argument("settings",help="Read in settings from this file.",type=str)
    parser_breakbox.set_defaults(func=dicer.dice)
    
    parser_stats = subparsers.add_parser('stats',help='Takes in a data file, computes statistics and makes plots.')
    parser_stats.add_argument("datafile",help='JSON file to read data from.',type=str)
    parser_stats.set_defaults(func=stats.statistics)

    parser_survey = subparsers.add_parser('survey',help='Takes in a survey file, bins it into a histogram and fits the survey function to it.')
    parser_survey.add_argument("settings",help='JSON file to read settings from.',type=str)
    parser_survey.set_defaults(func=survey.mainrun)

    args = parser.parse_args()

    function = None
    try:
        function =args.func
    except AttributeError:
        parser.print_help()
        exit()
    function(args)








if __name__ == "__main__":
    main()
