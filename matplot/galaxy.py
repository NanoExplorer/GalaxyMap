from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import argparse
import common
import correlation_func
import dicer
import stats
import survey
import surveytranspose
import surveystats
import jackknife
import velocity_correlation
import rawvcorr


def main():
    #Handle command line arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    #Generate settings file
    parser_gensettings = subparsers.add_parser('gensettings', help="generates a sample settings file")
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
    parser_selectsurvey.add_argument('-g','--gpu',help='use the GPU to try to speed up computations. GPU works best for large numbers of surveys.',action='store_true')
    parser_selectsurvey.set_defaults(func=survey.selectrun)
    
    parser_surveytranspose = subparsers.add_parser('transpose',help='Transforms cartesian surveys into CF2 format')
    parser_surveytranspose.add_argument('survey_file',help='survey.json file that contains a list of surveys and all of their center points.')
    parser_surveytranspose.set_defaults(func=surveytranspose.transpose)

    parser_velocity = subparsers.add_parser('vcorr',help='Computes the velocity correlations of a sample, according to the options in the settings file.')
    parser_velocity.add_argument("settings",help='JSON file to read settings from.',type=str)
    parser_velocity.set_defaults(func=velocity_correlation.main)

    add_parser(subparsers, 'rawvcorr', 'Computes the velocity correlations using full 3-dimensional velocities',
               [['settings']], ['The settings json file'], [str], rawvcorr.main)

    args = parser.parse_args()

    function = None
    try:
        function =args.func
    except AttributeError:
        print("Function not found")
        parser.print_help()
        exit()
    function(args)



def add_parser(subparser, moduleName, moduleHelp, argumentsList, argumentsHelpList, argumentsTypeList, defaultFunc):
    theParser = subparser.add_parser(moduleName, help = moduleHelp)
    for argument,theHelp,theType in zip(argumentsList, argumentsHelpList, argumentsTypeList):
        theParser.add_argument(*argument, help = theHelp, type = theType)
    theParser.set_defaults(func=defaultFunc)




if __name__ == "__main__":
    main()
