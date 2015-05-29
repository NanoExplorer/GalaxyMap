import argparse
import common
import correlation_func
import dicer
import stats
import survey
import surveytranspose
import surveystats
import jackknife

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

    parser_jackknife = subparsers.add_parser('jackknife', help="Takes in a box, divides it into subboxes containing randomly assigned points.")
    parser_jackknife.add_argument("settings",help="json file to read settings from.",type=str)
    parser_jackknife.set_defaults(func=jackknife.dice)

    parser_surveystats = subparsers.add_parser('surveystats',help='Takes in a survey file, bins it into a histogram and fits the survey function to it.')
    parser_surveystats.add_argument("settings",help='JSON file to read settings from.',type=str)
    parser_surveystats.set_defaults(func=surveystats.statsrun)

    parser_selectsurvey = subparsers.add_parser('select',help='Uses a settings file to extract lots of virtual surveys (like the cf2 and composite ones) from the huge millennium data file.')
    parser_selectsurvey.add_argument('settings',help='JSON file to read settings from.',type=str)
    parser_selectsurvey.set_defaults(func=survey.selectrun)

    parser_surveytranspose = subparsers.add_parser('transpose',help='Transforms cartesian surveys into CF2 format')
    parser_surveytranspose.add_argument('survey_file',help='survey.json file that contains a list of surveys and all of their center points.')
    parser_surveytranspose.set_defaults(func=surveytranspose.transpose)

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
