import sys
import argparse

from parser import build_ast, write_parser_output
from lexer import run_lexical_analysis
from type_checker import run_type_checker
from codegen import run_code_generator 

def display_version():
    """Prints the compiler version information as per Phase 1."""
    print("My own C compiler ")
    print("Written by Shrestha Banerjee (sb2001@iastate.edu)")
    print("Version 1.0, released 5 October, 2025")

def main():
    """Parses command line arguments and runs the correct compiler phase."""
    parser = argparse.ArgumentParser(
        description="A C compiler for the COMS 5400 project."
    )
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-1', dest='mode1', action='store_true', help="Phase 1: Version")
    mode_group.add_argument('-2', dest='mode2', action='store_true', help="Phase 2: Lexer")
    mode_group.add_argument('-3', dest='mode3', action='store_true', help="Phase 3: Parser")
    mode_group.add_argument('-4', dest='mode4', action='store_true', help="Phase 4: Type Checker")
    mode_group.add_argument('-5', dest='mode5', action='store_true', help="Phase 5: Code Generation")
    mode_group.add_argument('-6', dest='mode6', action='store_true', help="Phase 6: Code Generation (Full)")
    parser.add_argument('infile', nargs='?', default=None, help="The input source file.")

    args = parser.parse_args()

    if args.mode1:
        display_version()
    
    elif args.mode2:
        if not args.infile: sys.exit("Error: Input file required.")
        run_lexical_analysis(args.infile)
    
    elif args.mode3:
        if not args.infile: sys.exit("Error: Input file required.")
        ast = build_ast(args.infile)
        if ast: write_parser_output(ast, args.infile)
        else: sys.exit(1)

    elif args.mode4:
        if not args.infile: sys.exit("Error: Input file required.")
        ast = build_ast(args.infile)
        if ast:
            success = run_type_checker(args.infile, ast)
            if not success: sys.exit(1)
        else: sys.exit(1)

    elif args.mode5 or args.mode6: 
        if not args.infile: sys.exit("Error: Input file required.")
        ast = build_ast(args.infile)
        if ast:
            if run_type_checker(args.infile, ast):
                 run_code_generator(args.infile, ast)
            else:
                 sys.stderr.write("Type checking failed. Code generation aborted.\n")
                 sys.exit(1)
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()