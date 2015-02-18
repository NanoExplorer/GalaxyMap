import argparse
import common
import correlation_func
import dicer

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
